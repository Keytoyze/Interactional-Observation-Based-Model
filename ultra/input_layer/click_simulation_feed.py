"""Simulate click data based on human annotations.

See the following paper for more information on the simulation data.

    * Qingyao Ai, Keping Bi, Cheng Luo, Jiafeng Guo, W. Bruce Croft. 2018. Unbiased Learning to Rank with Unbiased Propensity Estimation. In Proceedings of SIGIR '18

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import json
import numpy as np
from ultra.input_layer import BaseInputFeed
from ultra.utils import click_models as cm
import ultra.utils

import tensorflow as tf
# We disable pylint because we need python3 compatibility.
from six.moves import zip  # pylint: disable=redefined-builtin


class ClickSimulationFeed(BaseInputFeed):
    """Simulate clicks based on human annotations.

    This class implements a input layer for unbiased learning to rank experiments
    by simulating click data based on both the human relevance annotation of
    each query-document pair and a predefined click model.
    """

    def __init__(self, model, batch_size, hparam_str, session=None):
        """Create the model.

        Args:
            model: (BasicModel) The model we are going to train.
            batch_size: the size of the batches generated in each iteration.
            hparam_str: the hyper-parameters for the input layer.
        """
        self.hparams = ultra.utils.hparams.HParams(
            # the setting file for the predefined click models.
            click_model_json='./example/ClickModel/pbm_0.1_1.0_4_1.0.json',
            # Set True to feed relevance labels instead of simulated clicks.
            oracle_mode=False,
            # Set eta change step for dynamic bias severity in training, 0.0
            # means no change.
            dynamic_bias_eta_change=0.0,
            # Set how many steps to change eta for dynamic bias severity in
            # training, 0.0 means no change.
            dynamic_bias_step_interval=1000,
            randomization=False,
            forward_only=False,
            context_w=None
        )

        print('Create simluated clicks feed')
        print(hparam_str)
        self.hparams.parse(hparam_str)
        self.click_model = None
        if not self.hparams.oracle_mode:
            with open(self.hparams.click_model_json) as fin:
                model_desc = json.load(fin)
                self.click_model = cm.loadModelFromJson(model_desc)
                self.context_w = np.array(model_desc['context_w'])
        self.context_eta_cache = {}

        self.start_index = 0
        self.count = 1
        if self.hparams.forward_only:
            self.rank_list_size = model.max_candidate_num
        else:
            self.rank_list_size = model.rank_list_size
        self.feature_size = model.feature_size
        self.batch_size = batch_size
        self.model = model
        self.global_batch_count = 0

    def prepare_sim_clicks_with_index(
            self, data_set, index, docid_inputs, letor_features, labels, true_labels, check_validation=True,
            randomization=False):
        i = index
        list_indices = list(range(self.rank_list_size))
        if randomization:
            random.shuffle(list_indices)

        # Generate clicks with click models.
        label_list = [
            0 if data_set.initial_list[i][x] < 0 else data_set.labels[i][x] for x in list_indices]
        # true_labels.append([0 if x > 0 else 1 for x in label_list])
        click_list = None
        if self.hparams.oracle_mode:
            click_list = label_list
            exam_p_list = [1] * len(label_list)
        else:
            self.click_model.dynamic_eta = self.get_context_eta(data_set, i)
            click_list, exam_p_list, _ = self.click_model.sampleClicksForOneList(
                list(label_list))
            # sample_count = 0
            # while check_validation and sum(click_list) == 0 and sample_count < self.MAX_SAMPLE_ROUND_NUM:
            #    click_list, _, _ = self.click_model.sampleClicksForOneList(list(label_list))
            #    sample_count += 1

        # Check if data is valid
        if check_validation:
            if sum(click_list) == 0:
                return
        base = len(letor_features)
        for x in list_indices:
            if data_set.initial_list[i][x] >= 0:
                letor_features.append(
                    data_set.features[data_set.initial_list[i][x]])
        docid_inputs.append(list([-1 if data_set.initial_list[i][x]
                                        < 0 else base + x for x in list_indices]))
        true_labels.append(label_list)
        labels.append(click_list)
        return exam_p_list

    def get_context_eta(self, data_set, i):
        if i in self.context_eta_cache:
            return self.context_eta_cache[i]
        feature_avg = np.zeros(len(self.context_w))
        feature_count = 0
        for x in data_set.initial_list[i]:
            if x < 0:
                continue
            feature_count += 1
            feature_avg += np.array(data_set.features[x])
        feature_avg = feature_avg / feature_count * self.context_w
        context_eta = max(0, 1 + sum(feature_avg))
        self.context_eta_cache[i] = context_eta
        return self.context_eta_cache[i]

    def get_batch(self, data_set, check_validation=False):
        """Get a random batch of data, prepare for step. Typically used for training.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            check_validation: (bool) Set True to ignore data with no positive labels.

        Returns:
            input_feed: a feed dictionary for the next step
            info_map: a dictionary contain some basic information about the batch (for debugging).

        """

        if len(data_set.initial_list[0]) < self.rank_list_size:
            raise ValueError("Input ranklist length must be no less than the required list size,"
                             " %d != %d." % (len(data_set.initial_list[0]), self.rank_list_size))
        length = len(data_set.initial_list)
        docid_inputs, letor_features, labels, true_labels = [], [], [], []
        rank_list_idxs = []
        batch_num = len(docid_inputs)
        exam_p_list = [[] for _ in range(self.rank_list_size)]
        while len(docid_inputs) < self.batch_size:
            i = int(random.random() * length)
            _exam_p_list = self.prepare_sim_clicks_with_index(data_set, i,
                                                              docid_inputs, letor_features, labels,
                                                              true_labels,
                                                              check_validation,
                                                              self.hparams.randomization)
            if _exam_p_list is not None:
                _exam_p_list = [_exam_p_list[0] / (x + 1e-7) for x in _exam_p_list]
                for p in range(self.rank_list_size):
                    exam_p_list[p].append(_exam_p_list[p])
            if batch_num < len(docid_inputs):  # new list added
                rank_list_idxs.append(i)
                batch_num = len(docid_inputs)
        local_batch_size = len(docid_inputs)
        letor_features_length = len(letor_features)
        for i in range(local_batch_size):
            for j in range(self.rank_list_size):
                if docid_inputs[i][j] < 0:
                    docid_inputs[i][j] = letor_features_length

        batch_docid_inputs = []
        batch_labels = []
        batch_true_labels = []
        for length_idx in range(self.rank_list_size):
            # Batch encoder inputs are just re-indexed docid_inputs.
            batch_docid_inputs.append(
                np.array([docid_inputs[batch_idx][length_idx]
                          for batch_idx in range(local_batch_size)], dtype=np.float32))
            # Batch decoder inputs are re-indexed decoder_inputs, we create
            # labels.
            batch_labels.append(
                np.array([labels[batch_idx][length_idx]
                          for batch_idx in range(local_batch_size)], dtype=np.float32))
            batch_true_labels.append(
                np.array([true_labels[batch_idx][length_idx]
                          for batch_idx in range(local_batch_size)], dtype=np.float32))
        # Create input feed map
        input_feed = {}
        input_feed[self.model.letor_features.name] = np.array(letor_features)
        for l in range(self.rank_list_size):
            input_feed[self.model.docid_inputs[l].name] = batch_docid_inputs[l]
            input_feed[self.model.labels[l].name] = batch_labels[l]
            input_feed[self.model.relevances[l].name] = batch_true_labels[l]
            input_feed[self.model.exam_p_list[l].name] = np.asarray(exam_p_list[l], dtype=np.float32)
        # Create info_map to store other information
        info_map = {
            'rank_list_idxs': rank_list_idxs,
            'input_list': docid_inputs,
            'click_list': labels,
            'letor_features': letor_features
        }

        self.global_batch_count += 1
        if self.hparams.dynamic_bias_eta_change != 0 and not self.hparams.oracle_mode:
            if self.global_batch_count % self.hparams.dynamic_bias_step_interval == 0:
                self.click_model.eta += self.hparams.dynamic_bias_eta_change
                self.click_model.setExamProb(self.click_model.eta)
                print(
                    'Dynamically change bias severity eta to %.3f' %
                    self.click_model.eta)

        return input_feed, info_map

    def get_next_batch(self, index, data_set, check_validation=False):
        """Get the next batch of data from a specific index, prepare for step.
           Typically used for validation.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

        Args:
            index: the index of the data before which we will use to create the data batch.
            data_set: (Raw_data) The dataset used to build the input layer.
            check_validation: (bool) Set True to ignore data with no positive labels.

        Returns:
            input_feed: a feed dictionary for the next step
            info_map: a dictionary contain some basic information about the batch (for debugging).

        """
        if len(data_set.initial_list[0]) < self.rank_list_size:
            raise ValueError("Input ranklist length must be no less than the required list size,"
                             " %d != %d." % (len(data_set.initial_list[0]), self.rank_list_size))

        docid_inputs, letor_features, labels, true_labels = [], [], [], []

        num_remain_data = len(data_set.initial_list) - index
        exam_p_list = [[] for _ in range(self.rank_list_size)]
        for offset in range(min(self.batch_size, num_remain_data)):
            i = index + offset
            _exam_p_list = self.prepare_sim_clicks_with_index(
                data_set, i, docid_inputs, letor_features, labels, true_labels, check_validation)
            if _exam_p_list is not None:
                _exam_p_list = [_exam_p_list[0] / (x + 1e-7) for x in _exam_p_list]
                for p in range(self.rank_list_size):
                    exam_p_list[p].append(_exam_p_list[p])

        local_batch_size = len(docid_inputs)
        letor_features_length = len(letor_features)
        for i in range(local_batch_size):
            for j in range(self.rank_list_size):
                if docid_inputs[i][j] < 0:
                    docid_inputs[i][j] = letor_features_length

        batch_docid_inputs = []
        batch_labels = []
        batch_true_labels = []
        for length_idx in range(self.rank_list_size):
            # Batch encoder inputs are just re-indexed docid_inputs.
            batch_docid_inputs.append(
                np.array([docid_inputs[batch_idx][length_idx]
                          for batch_idx in range(local_batch_size)], dtype=np.float32))
            # Batch decoder inputs are re-indexed decoder_inputs, we create
            # weights.
            batch_labels.append(
                np.array([labels[batch_idx][length_idx]
                          for batch_idx in range(local_batch_size)], dtype=np.float32))
            batch_true_labels.append(
                np.array([true_labels[batch_idx][length_idx]
                          for batch_idx in range(local_batch_size)], dtype=np.float32))
        # Create input feed map
        input_feed = {}
        input_feed[self.model.letor_features.name] = np.array(letor_features)
        for l in range(self.rank_list_size):
            input_feed[self.model.docid_inputs[l].name] = batch_docid_inputs[l]
            input_feed[self.model.labels[l].name] = batch_labels[l]
            input_feed[self.model.exam_p_list[l].name] = np.asarray(exam_p_list[l])
            input_feed[self.model.relevances[l].name] = batch_true_labels[l]
        # Create others_map to store other information
        others_map = {
            'input_list': docid_inputs,
            'click_list': labels,
        }
        return input_feed, others_map

    def get_data_by_index(self, data_set, index, check_validation=False):
        """Get one data from the specified index, prepare for step.

                Args:
                    data_set: (Raw_data) The dataset used to build the input layer.
                    index: the index of the data
                    check_validation: (bool) Set True to ignore data with no positive labels.

                Returns:
                    The triple (docid_inputs, decoder_inputs, target_weights) for
                    the constructed batch that has the proper format to call step(...) later.
                """
        if len(data_set.initial_list[0]) < self.rank_list_size:
            raise ValueError("Input ranklist length must be no less than the required list size,"
                             " %d != %d." % (len(data_set.initial_list[0]), self.rank_list_size))

        docid_inputs, letor_features, labels = [], [], []

        i = index
        self.prepare_sim_clicks_with_index(
            data_set,
            i,
            docid_inputs,
            letor_features,
            labels,
            check_validation)

        letor_features_length = len(letor_features)
        for j in range(self.rank_list_size):
            if docid_inputs[-1][j] < 0:
                docid_inputs[-1][j] = letor_features_length

        batch_docid_inputs = []
        batch_labels = []
        for length_idx in range(self.rank_list_size):
            # Batch encoder inputs are just re-indexed docid_inputs.
            batch_docid_inputs.append(
                np.array([docid_inputs[batch_idx][length_idx]
                          for batch_idx in range(1)], dtype=np.float32))
            # Batch decoder inputs are re-indexed decoder_inputs, we create
            # weights.
            batch_labels.append(
                np.array([labels[batch_idx][length_idx]
                          for batch_idx in range(1)], dtype=np.float32))
        # Create input feed map
        input_feed = {}
        input_feed[self.model.letor_features.name] = np.array(letor_features)
        for l in range(self.rank_list_size):
            input_feed[self.model.docid_inputs[l].name] = batch_docid_inputs[l]
            input_feed[self.model.labels[l].name] = batch_labels[l]
        # Create others_map to store other information
        others_map = {
            'input_list': docid_inputs,
            'click_list': labels,
        }

        return input_feed, others_map
