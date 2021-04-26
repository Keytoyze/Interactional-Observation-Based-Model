"""The basic class that contains all the API needed for the implementation of an unbiased learning to rank algorithm.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np

import ultra.utils as utils


class BaseAlgorithm(ABC):
    """The basic class that contains all the API needed for the
        implementation of an unbiased learning to rank algorithm.

    """
    PADDING_SCORE = -100000

    @abstractmethod
    def __init__(self, data_set, exp_settings, forward_only=False):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
            forward_only: Set true to conduct prediction only, false to conduct training.
        """
        self.is_training = None
        self.docid_inputs = None  # a list of top documents
        self.letor_features = None  # the letor features for the documents
        self.labels = None  # the labels for the documents (e.g., clicks)
        self.output = None  # the ranking scores of the inputs
        # the number of documents considered in each rank list.
        self.rank_list_size = None
        # the maximum number of candidates for each query.
        self.max_candidate_num = None
        self.optimizer_func = tf.train.AdagradOptimizer
        pass

    @abstractmethod
    def step(self, session, input_feed, forward_only):
        """Run a step of the model feeding the given inputs.

        Args:
            session: (tf.Session) tensorflow session to use.
            input_feed: (dictionary) A dictionary containing all the input feed data.
            forward_only: whether to do the backward step (False) or only forward (True).

        Returns:
            A triple consisting of the loss, outputs (None if we do backward),
            and a tf.summary containing related information about the step.

        """
        pass

    def remove_padding_for_metric_eval(self, input_id_list, model_output):
        output_scores = tf.unstack(model_output, axis=1)
        if len(output_scores) > len(input_id_list):
            raise AssertionError(
                'Input id list is shorter than output score list when remove padding.')
        # Build mask
        valid_flags = tf.cast(
            tf.concat(
                values=[tf.ones([tf.shape(self.letor_features)[0]]), tf.zeros([1])], axis=0),
            tf.bool
        )
        input_flag_list = []
        for i in range(len(output_scores)):
            input_flag_list.append(
                tf.nn.embedding_lookup(
                    valid_flags, input_id_list[i]))
        # Mask padding documents
        for i in range(len(output_scores)):
            output_scores[i] = tf.where(
                input_flag_list[i],
                output_scores[i],
                tf.ones_like(output_scores[i]) * self.PADDING_SCORE
            )
        return tf.stack(output_scores, axis=1)

    def ranking_model(self, list_size, scope=None):
        """Construct ranking model with the given list size.

        Args:
            list_size: (int) The top number of documents to consider in the input docids.
            scope: (string) The name of the variable scope.

        Returns:
            A tensor with the same shape of input_docids.

        """
        context_embedding, output_scores = self.get_ranking_scores(
            self.docid_inputs[:list_size], self.is_training, scope)
        return context_embedding, tf.concat(output_scores, 1)

    def get_propensity_scores(self, is_training, click_label):
        with tf.variable_scope("propensity_model"):
            # If we are in testing, do not compute propensity
            if not is_training:
                return tf.ones_like(click_label)
            if not hasattr(self, "propensity_model") or self.propensity_model is None:
                if 'propensity_model' not in self.exp_settings:
                    self.exp_settings['propensity_model'] = "ultra.propensity_model.PBM"
                    self.exp_settings['propensity_model_hparams'] = ""
                self.propensity_model = utils.find_class(
                    self.exp_settings['propensity_model'])(
                    self.exp_settings['propensity_model_hparams'])
            result = self.propensity_model.build(is_training, click_label, self)
            size = lambda v: np.prod(v.get_shape().as_list())
            n = sum(size(v) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "(ranking_model/)?propensity_model"))
            print([v.name for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "(ranking_model/)?propensity_model")])
            print("Propensity model size: %d" % (n))
            return result

    def get_ranking_scores(self, input_id_list,
                           is_training=False, scope=None, **kwargs):
        """Compute ranking scores with the given inputs.

        Args:
            input_id_list: (list<tf.Tensor>) A list of tensors containing document ids.
                            Each tensor must have a shape of [None].
            is_training: (bool) A flag indicating whether the model is running in training mode.
            scope: (string) The name of the variable scope.

        Returns:
            A tensor with the same shape of input_docids.

        """
        with tf.variable_scope(scope or "ranking_model"):
            # Build feature padding
            PAD_embed = tf.zeros([1, self.feature_size], dtype=tf.float32)
            letor_features = tf.concat(
                axis=0, values=[
                    self.letor_features, PAD_embed])
            input_feature_list = []
            if not hasattr(self, "model") or self.model is None:
                self.model = utils.find_class(
                    self.exp_settings['ranking_model'])(
                    self.exp_settings['ranking_model_hparams'])
            for i in range(len(input_id_list)):
                input_feature_list.append(
                    tf.nn.embedding_lookup(
                        letor_features, input_id_list[i]))
            return self.model.build(
                input_feature_list, is_training=is_training, **kwargs)

    def pairwise_cross_entropy_loss(self, pos_scores, neg_scores, name=None):
        """Computes pairwise softmax loss without propensity weighting.

        Args:
            pos_scores: (tf.Tensor) A tensor with shape [batch_size, 1]. Each value is
            the ranking score of a positive example.
            neg_scores: (tf.Tensor) A tensor with shape [batch_size, 1]. Each value is
            the ranking score of a negative example.
            name: A string used as the name for this variable scope.

        Returns:
            (tf.Tensor) A single value tensor containing the loss.
        """
        loss = None
        with tf.name_scope(name, "pairwise_cross_entropy_loss", [pos_scores, neg_scores]):
            label_dis = tf.concat(
                [tf.ones_like(pos_scores), tf.zeros_like(neg_scores)], axis=1)
            loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=tf.concat([pos_scores, neg_scores], axis=1), labels=label_dis
            )
        return loss
