"""Training and testing the dual learning algorithm for unbiased learning to rank.

See the following paper for more information on the dual learning algorithm.

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
import numpy as np
import tensorflow as tf

import copy
import itertools
from six.moves import zip
from tensorflow import dtypes
from ultra.learning_algorithm.base_algorithm import BaseAlgorithm
import ultra.utils


def sigmoid_prob(logits):
    # return tf.sigmoid(logits - tf.reduce_mean(logits, -1, keep_dims=True))
    return tf.sigmoid(logits)
    # return (tf.tanh(logits) + 1) / 2


class DLA(BaseAlgorithm):
    """The Dual Learning Algorithm for unbiased learning to rank.

    This class implements the Dual Learning Algorithm (DLA) based on the input layer
    feed. See the following paper for more information on the algorithm.

    * Qingyao Ai, Keping Bi, Cheng Luo, Jiafeng Guo, W. Bruce Croft. 2018. Unbiased Learning to Rank with Unbiased Propensity Estimation. In Proceedings of SIGIR '18

    """

    def get_loss(self, loss_name):
        if loss_name == 'click_weighted_softmax_cross_entropy':
            return self.click_weighted_softmax_cross_entropy_loss
        elif loss_name == 'click_weighted_log_loss':
            return self.click_weighted_log_loss
        elif loss_name == 'click_weighted_pairwise_loss':
            return self.click_weighted_pairwise_loss
        elif loss_name == 'click_weighted_log_ratio_loss':
            return self.click_weighted_log_ratio_loss
        elif loss_name == 'click_weighted_mse_loss':
            return self.click_weighted_log_loss
        elif loss_name == 'click_weighted_softmax_cross_entropy_point_loss':
            return self.click_weighted_softmax_cross_entropy_point_loss
        elif loss_name == 'click_weighted_log_loss_1':
            return self.click_weighted_log_loss_1
        elif loss_name == 'pairwise_debias_propensity_loss':
            return self.pairwise_debias_propensity_loss
        elif loss_name == 'pairwise_debias_ranking_loss':
            return self.pairwise_debias_ranking_loss
        else:  # softmax loss without weighting
            return self.softmax_loss

    def get_optimizer(self, optimizer):
        if optimizer == 'sgd':
            return tf.train.GradientDescentOptimizer
        elif optimizer == 'adam':
            return tf.train.AdamOptimizer
        elif optimizer == 'radam':
            return ultra.utils.RAdamOptimizer
        return tf.train.AdagradOptimizer

    def get_logits_to_prob(self, loss_name):
        if loss_name == 'click_weighted_log_loss' or loss_name == 'click_weighted_log_ratio_loss' or loss_name == 'click_weighted_log_loss_1':
            return lambda x: x
        else:
            return tf.nn.softmax

    def pre_train_filter(self, weight):
        # w = tf.cast(tf.clip_by_value(
        #     (self.hparams.pre_train_step - self.global_step) / self.hparams.pre_train_step,
        #     0, 1
        # ), dtype=tf.float32)
        ones = tf.ones_like(weight)
        return tf.cond(self.global_step <= self.hparams.pre_train_step,
                       lambda: ones,  # w * ones + (1 - w) * weight,
                       lambda: weight)

    def __init__(self, data_set, exp_settings, forward_only=False):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
            forward_only: Set true to conduct prediction only, false to conduct training.
        """
        print('Build DLA')

        self.hparams = ultra.utils.hparams.HParams(
            learning_rate=0.05,  # Learning rate.
            max_gradient_norm=5.0,  # Clip gradients to this norm.
            loss_func='click_weighted_softmax_cross_entropy',  # Select Loss function
            propensity_loss_func='click_weighted_softmax_cross_entropy',
            # the function used to convert logits to probability distributions
            logits_to_prob='softmax',
            # The learning rate for ranker (-1 means same with learning_rate).
            propensity_learning_rate=-1.0,
            ranker_loss_weight=1.0,  # Set the weight of unbiased ranking loss
            # Set strength for L2 regularization.
            l2_loss=0.0,
            propensity_l2_loss=0.0,
            max_propensity_weight=-1,  # Set maximum value for propensity weights
            constant_propensity_initialization=False,
            # Set true to initialize propensity with constants.
            grad_strategy='sgd',  # Select gradient strategy
            propensity_grad_strategy=None,  # Select gradient strategy
            pre_train_step=-1,
            oracle=False
        )
        print(exp_settings)
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.exp_settings = exp_settings
        self.model = None
        self.max_candidate_num = exp_settings['max_candidate_num']
        self.feature_size = data_set.feature_size
        if self.hparams.propensity_learning_rate < 0:
            self.propensity_learning_rate = tf.Variable(
                float(self.hparams.learning_rate), trainable=False)
        else:
            self.propensity_learning_rate = tf.Variable(
                float(self.hparams.propensity_learning_rate), trainable=False)
        self.learning_rate = tf.Variable(
            float(self.hparams.learning_rate), trainable=False)

        # Feeds for inputs.
        self.is_training = tf.placeholder(tf.bool, name="is_train")
        self.docid_inputs = []  # a list of top documents
        self.letor_features = tf.placeholder(tf.float32, shape=[None, self.feature_size],
                                             name="letor_features")  # the letor features for the documents
        self.labels = []  # the labels for the documents (e.g., clicks)
        self.relevances = []
        # Log real propensity
        self.exam_p_list = []
        for i in range(self.max_candidate_num):
            self.docid_inputs.append(tf.placeholder(tf.int64, shape=[None],
                                                    name="docid_input{0}".format(i)))
            self.labels.append(tf.placeholder(tf.float32, shape=[None],
                                              name="label{0}".format(i)))
            self.relevances.append(tf.placeholder(tf.float32, shape=[None],
                                                  name="relevance{0}".format(i)))
            self.exam_p_list.append(tf.placeholder(tf.float32, shape=[None],
                                                   name="exam_p{0}".format(i)))

        self.global_step = tf.Variable(0, trainable=False)

        # Select logits to prob function
        self.rank_logits_to_prob = self.get_logits_to_prob(self.hparams.loss_func)
        self.propensity_logits_to_prob = self.get_logits_to_prob(self.hparams.propensity_loss_func)

        _, self.output = self.ranking_model(
            self.max_candidate_num, scope='ranking_model')
        self.context_embedding, self.train_output = self.ranking_model(
            exp_settings['selection_bias_cutoff'], scope='ranking_model')
        self.relevance_weights = self.get_normalized_weights(
            self.rank_logits_to_prob(self.train_output))
        pad_removed_output = self.remove_padding_for_metric_eval(
            self.docid_inputs, self.output)
        # reshape from [max_candidate_num, ?] to [?, max_candidate_num]
        reshaped_relevance = tf.transpose(tf.convert_to_tensor(self.relevances))
        reshaped_relevance_cut = tf.transpose(
            tf.convert_to_tensor(self.relevances[:exp_settings['selection_bias_cutoff']]))
        for metric in self.exp_settings['metrics']:
            for topn in self.exp_settings['metrics_topn']:
                metric_value = ultra.utils.make_ranking_metric_fn(
                    metric, topn)(reshaped_relevance, pad_removed_output, None)
                tf.summary.scalar(
                    '%s_%d' %
                    (metric, topn), metric_value, collections=['eval'])

        # Eval IPS
        if self.hparams.oracle:
            print("Note: use oracle model!!!")
            train_labels = self.relevances[:exp_settings['selection_bias_cutoff']]
        else:
            train_labels = self.labels[:exp_settings['selection_bias_cutoff']]
        reshaped_train_labels = tf.transpose(tf.convert_to_tensor(train_labels))
        self.propensity = self.get_propensity_scores(True, reshaped_train_labels)
        self.propensity_weights = self.get_normalized_weights(self.propensity_logits_to_prob(self.propensity))
        pw_list = tf.unstack(
            self.propensity_weights,
            axis=1)  # Compute propensity weights

        epsilon = 1e-7
        ips_mse = tf.losses.mean_squared_error(tf.stack(pw_list),
                                            tf.stack(self.exam_p_list[:len(pw_list)]))
        ips_error = tf.abs(tf.stack([(
                                            (pw_list[i] - self.exam_p_list[i]) /
                                            (self.exam_p_list[i] * math.log1p(i) + epsilon)
                                    ) if i != 0 else tf.zeros_like(pw_list[i]) for i in
                                    range(len(pw_list))]))
        # ips_error_mean, ips_error_var = tf.nn.moments(ips_error, axes=[1])
        for i in range(len(pw_list)):
            tf.summary.scalar(
                'IPS_%d' % i, tf.reduce_mean(pw_list[i]), collections=['eval'])
        tf.summary.scalar('IPS_MSE', tf.reduce_mean(ips_mse), collections=['eval'])
        tf.summary.scalar('IPS_D_ERROR', tf.reduce_mean(ips_error), collections=['eval'])

        norm_exam_p_list = self.normalize_p_list(self.exam_p_list[:len(pw_list)])
        norm_pw_list = self.normalize_p_list(pw_list)
        kl = tf.reduce_sum(norm_exam_p_list * (
                tf.log(norm_exam_p_list + epsilon) - tf.log(norm_pw_list + epsilon)
        ), axis=0)
        tf.summary.scalar('KL', tf.reduce_mean(kl), collections=['eval'])
        self.selection_bias_cutoff = exp_settings['selection_bias_cutoff']

        if not forward_only:
            # Build model
            self.rank_list_size = exp_settings['selection_bias_cutoff']

            print('Loss Function is ' + self.hparams.loss_func)
            # Select loss function
            self.loss_func = self.get_loss(self.hparams.loss_func)
            self.propensity_loss_func = self.get_loss(self.hparams.propensity_loss_func)

            # Compute rank loss
            tf.summary.scalar(
                'Click_rate', tf.reduce_mean(reshaped_train_labels), collections=['eval'])
            self.rank_loss = self.loss_func(
                self.train_output, reshaped_train_labels, self.pre_train_filter(self.propensity_weights))
            tf.summary.scalar(
                'Rank_Loss',
                tf.reduce_mean(self.rank_loss),
                collections=['train', 'eval'])

            # Compute examination loss
            self.exam_loss = self.propensity_loss_func(
                self.propensity,
                reshaped_train_labels,
                self.relevance_weights)
            rw_list = tf.unstack(
                self.relevance_weights,
                axis=1)  # Compute propensity weights
            for i in range(len(rw_list)):
                tf.summary.scalar(
                    'Relevance weights %d' %
                    i, tf.reduce_mean(
                        rw_list[i]), collections=['train'])
            tf.summary.scalar(
                'Exam Loss',
                tf.reduce_mean(
                    self.exam_loss),
                collections=['train', 'eval'])

            # Gradients and SGD update operation for training the model.
            self.loss = self.exam_loss + self.hparams.ranker_loss_weight * self.rank_loss

            # Select optimizer
            self.optimizer_func = self.get_optimizer(self.hparams.grad_strategy)
            if self.hparams.propensity_grad_strategy is None:
                self.propensity_optimizer_func = self.optimizer_func
            else:
                self.propensity_optimizer_func = self.get_optimizer(self.hparams.propensity_grad_strategy)

            self.separate_gradient_update()

            tf.summary.scalar(
                'Gradient Norm',
                self.norm,
                collections=['train'])
            tf.summary.scalar(
                'Learning Rate',
                self.learning_rate,
                collections=['train'])
            tf.summary.scalar(
                'Final Loss', tf.reduce_mean(
                    self.loss), collections=['train'])

            clipped_labels = tf.clip_by_value(
                reshaped_train_labels, clip_value_min=0, clip_value_max=1)
            pad_removed_train_output = self.remove_padding_for_metric_eval(
                self.docid_inputs, self.train_output)
            for metric in self.exp_settings['metrics']:
                for topn in self.exp_settings['metrics_topn']:
                    list_weights = tf.reduce_mean(
                        self.propensity_weights * clipped_labels, axis=1, keep_dims=True)
                    metric_value = ultra.utils.make_ranking_metric_fn(metric, topn)(
                        reshaped_train_labels, pad_removed_train_output, None)
                    tf.summary.scalar(
                        '%s_%d' %
                        (metric, topn), metric_value, collections=['train'])
                    weighted_metric_value = ultra.utils.make_ranking_metric_fn(metric, topn)(
                        reshaped_train_labels, pad_removed_train_output, list_weights)
                    tf.summary.scalar(
                        'Weighted_%s_%d' %
                        (metric, topn), weighted_metric_value, collections=['train'])
        else:
            self.rank_list_size = exp_settings['max_candidate_num']

        self.train_summary = tf.summary.merge_all(key='train')
        self.eval_summary = tf.summary.merge_all(key='eval')
        self.saver = tf.train.Saver(tf.global_variables())

    def normalize_p_list(self, lst):
        lst = tf.stack(lst)
        return lst / tf.reduce_sum(lst, axis=0, keepdims=True)

    def separate_gradient_update(self):
        self.denoise_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "(ranking_model/)?propensity_model")
        self.ranking_model_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "ranking_model")

        if self.hparams.l2_loss > 0:
            for p in self.ranking_model_params:
                if 'kernel' in p.name:
                    self.rank_loss += self.hparams.l2_loss * tf.nn.l2_loss(p)
                    print("Add L2 regularization loss: " + p.name)                
        if self.hparams.propensity_l2_loss > 0:
            for p in self.denoise_params:
                if 'kernel' in p.name:
                    self.exam_loss += self.hparams.propensity_l2_loss * tf.nn.l2_loss(p)
                    print("Add L2 regularization loss: " + p.name)

        self.loss = self.exam_loss + self.hparams.ranker_loss_weight * self.rank_loss

        denoise_gradients = tf.gradients(self.exam_loss, self.denoise_params)
        ranking_model_gradients = tf.gradients(
            self.rank_loss, self.ranking_model_params)
        if self.hparams.max_gradient_norm > 0:
            denoise_gradients, denoise_norm = tf.clip_by_global_norm(denoise_gradients,
                                                                     self.hparams.max_gradient_norm)
            ranking_model_gradients, ranking_model_norm = tf.clip_by_global_norm(
                ranking_model_gradients,
                self.hparams.max_gradient_norm * self.hparams.ranker_loss_weight)
        self.norm = tf.global_norm(denoise_gradients + ranking_model_gradients)

        self.opt_denoise = self.propensity_optimizer_func(self.propensity_learning_rate)
        self.opt_ranker = self.optimizer_func(self.learning_rate)

        ranker_updates = self.opt_ranker.apply_gradients(
            zip(ranking_model_gradients, self.ranking_model_params))
        denoise_updates = self.opt_denoise.apply_gradients(
            zip(denoise_gradients, self.denoise_params),
            global_step=self.global_step)
        self.updates = tf.cond(
            tf.equal(self.propensity_learning_rate, 0),
            lambda: ranker_updates,
            lambda: tf.group(denoise_updates, ranker_updates))

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

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            # reset ranking optimizer if pre-train finishes
            if self.hparams.pre_train_step + 1 == self.global_step.eval():
                reset_optimizer_op = tf.variables_initializer(self.opt_ranker.variables())
                session.run(reset_optimizer_op)
            input_feed[self.is_training.name] = True
            output_feed = [self.updates,  # Update Op that does SGD.
                           self.loss,  # Loss for this batch.
                           self.train_summary  # Summarize statistics.
                           ]
        else:
            input_feed[self.is_training.name] = False
            output_feed = [
                self.eval_summary,  # Summarize statistics.
                self.output  # Model outputs
            ]

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            # loss, no outputs, summary.
            return outputs[1], None, outputs[-1]
        else:
            return None, outputs[1], outputs[0]  # no loss, outputs, summary.

    def softmax_loss(self, output, labels, propensity=None, name=None):
        """Computes listwise softmax loss without propensity weighting.

        Args:
            output: (tf.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (tf.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
            relevant example.
            propensity: No use.
            name: A string used as the name for this variable scope.

        Returns:
            (tf.Tensor) A single value tensor containing the loss.
        """

        loss = None
        with tf.name_scope(name, "softmax_loss", [output]):
            label_dis = labels / tf.reduce_sum(labels, 1, keep_dims=True)
            loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=output, labels=label_dis) * tf.reduce_sum(labels, 1)
        return tf.reduce_sum(loss) / tf.reduce_sum(labels)

    def get_normalized_weights(self, propensity):
        """Computes listwise softmax loss with propensity weighting.

        Args:
            propensity: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (tf.Tensor) A tensor containing the propensity weights.
        """
        propensity = tf.stop_gradient(propensity)
        propensity_list = tf.unstack(
            propensity, axis=1)  # Compute propensity weights
        pw_list = []
        for i in range(len(propensity_list)):
            pw_i = propensity_list[0] / propensity_list[i]
            pw_list.append(pw_i)
        propensity_weights = tf.stack(pw_list, axis=1)
        if self.hparams.max_propensity_weight > 0:
            propensity_weights = tf.clip_by_value(
                propensity_weights,
                clip_value_min=0,
                clip_value_max=self.hparams.max_propensity_weight)
        return tf.stop_gradient(propensity_weights)

    def click_weighted_softmax_cross_entropy_loss(
            self, output, labels, propensity_weights, name=None):
        """Computes listwise softmax loss with propensity weighting.

        Args:
            output: (tf.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (tf.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
            relevant example.
            propensity_weights: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.
            name: A string used as the name for this variable scope.

        Returns:
            (tf.Tensor) A single value tensor containing the loss.
        """
        loss = None
        with tf.name_scope(name, "click_softmax_cross_entropy", [output]):
            label_dis = labels * propensity_weights / \
                        (tf.reduce_sum(labels * propensity_weights, 1, keep_dims=True) + 1e-7)
            loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=output, labels=label_dis) * tf.reduce_sum(labels * propensity_weights, 1)
        return tf.reduce_sum(loss) / (tf.reduce_sum(labels * propensity_weights) + 1e-7)
    
    def click_weighted_softmax_cross_entropy_point_loss(
            self, output, labels, propensity_weights, name=None):
        """Computes listwise softmax loss with propensity weighting, without gradient propagation.

        Args:
            output: (tf.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (tf.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
            relevant example.
            propensity_weights: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.
            name: A string used as the name for this variable scope.

        Returns:
            (tf.Tensor) A single value tensor containing the loss.
        """
        loss = None
        with tf.name_scope(name, "click_softmax_cross_entropy_point", [output]):
            label_dis = labels * propensity_weights / \
                        (tf.reduce_sum(labels * propensity_weights, 1, keep_dims=True) + 1e-7)
            exp_output = tf.exp(output)
            softmax_denominator = tf.reduce_sum(exp_output, axis=-1, keepdims=True) + 1e-7
            softmax_denominator = tf.stop_gradient(softmax_denominator)
            softmax_output = exp_output / softmax_denominator + 1e-7
            loss = tf.keras.losses.categorical_crossentropy(label_dis, softmax_output)
            loss = loss * tf.reduce_sum(labels * propensity_weights, 1)
        return tf.reduce_sum(loss) / (tf.reduce_sum(labels * propensity_weights) + 1e-7)

    def click_weighted_pairwise_loss(
            self, output, labels, propensity_weights, name=None):
        """Computes pairwise entropy loss with propensity weighting.

        Args:
            output: (tf.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (tf.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
                relevant example.
            propensity_weights: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.
            name: A string used as the name for this variable scope.

        Returns:
            (tf.Tensor) A single value tensor containing the loss.
            (tf.Tensor) A tensor containing the propensity weights.
        """
        loss = None
        with tf.name_scope(name, "click_weighted_pairwise_loss", [output]):
            sliced_output = tf.unstack(output, axis=1)
            sliced_label = tf.unstack(labels, axis=1)
            sliced_propensity = tf.unstack(propensity_weights, axis=1)
            # delta NDCG
            inverse_idcg = tf.reshape(ultra.utils.inverse_max_dcg(labels), (-1, )) # (B, )
            for i in range(len(sliced_output)):
                for j in range(i + 1, len(sliced_output)):

                    s1 = sliced_output[i] # (B, )
                    s2 = sliced_output[j] # (B, )
                    s12 = tf.sigmoid(s1 - s2) # (B, ), 0~1
                    click1 = sliced_label[i] # (B, )
                    click2 = sliced_label[j] # (B, )
                    cur_labels = click1 - click2 # (B, ), 0/1/-1
                    weights = tf.abs(cur_labels) # (B, ), 0/1
                    cur_labels = (cur_labels + 1) / 2 # (B, ), 0/1

                    pos_loginv = 1.0 / math.log2(i + 2)
                    neg_loginv = 1.0 / math.log2(j + 2)
                    original = click1 * pos_loginv + click2 * neg_loginv
                    changed = click2 * pos_loginv + click1 * neg_loginv
                    delta = tf.abs((original - changed) * inverse_idcg) # (B, )

                    cur_click_propensity = (sliced_propensity[i] * click1 + sliced_propensity[j] * click2)
                    cur_unclick_propensity = (sliced_propensity[i] * (1 - click1) + sliced_propensity[j] * (1 - click2))
                    # cur_propensity = cur_click_propensity / (cur_unclick_propensity + 1e-5)
                    cur_propensity = cur_click_propensity
                    cur_pair_loss = tf.losses.log_loss(labels=cur_labels, predictions=s12, weights=weights * cur_propensity * delta)
                    
                    # cur_pair_loss = - \
                    #                     tf.exp(
                    #                         sliced_output[i]) / (
                    #                         tf.exp(sliced_output[i]) + tf.exp(
                    #                     sliced_output[j]) + 1e-7)

                    
                    if loss is None:
                        loss = cur_pair_loss
                    loss += cur_pair_loss
        batch_size = tf.shape(labels[0])[0]
        # / (tf.reduce_sum(propensity_weights)+1)
        return tf.reduce_sum(loss) / tf.cast(batch_size, dtypes.float32)

    def click_weighted_log_loss(
            self, output, labels, propensity_weights, name=None):
        """Computes pointwise sigmoid loss with propensity weighting.

        Args:
            output: (tf.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (tf.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
            relevant example.
            propensity_weights: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.
            name: A string used as the name for this variable scope.

        Returns:
            (tf.Tensor) A single value tensor containing the loss.
        """
        loss = None
        with tf.name_scope(name, "click_weighted_log_loss", [output]):
            target = labels * propensity_weights
            
            # target = tf.transpose(tf.convert_to_tensor(self.exam_p_list[:10]))
            # target = tf.ones_like(target) / target
            # tf.summary.scalar(
            #     'target', tf.reduce_mean(target), collections=['eval'])
            # tf.summary.scalar(
            #     'target_max', tf.reduce_max(target), collections=['eval'])
            loss = tf.losses.log_loss(target, output)
        return loss

    def click_weighted_log_loss_1(
            self, output, labels, propensity_weights, name=None):
        """Computes pointwise sigmoid loss with propensity weighting.

        Args:
            output: (tf.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (tf.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
            relevant example.
            propensity_weights: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.
            name: A string used as the name for this variable scope.

        Returns:
            (tf.Tensor) A single value tensor containing the loss.
        """
        loss = None
        with tf.name_scope(name, "click_weighted_log_loss1", [output]):
            target = labels * propensity_weights
            
            # target = tf.transpose(tf.convert_to_tensor(self.exam_p_list[:10]))
            # target = tf.ones_like(target) / target
            # tf.summary.scalar(
            #     'target', tf.reduce_mean(target), collections=['eval'])
            # tf.summary.scalar(
            #     'target_max', tf.reduce_max(target), collections=['eval'])
            weight = 1
            loss = tf.losses.log_loss(target, output)
        return loss
    
    def huber_loss(self, labels, predictions, delta=100.0):
        residual = tf.abs(predictions - labels)
        return tf.where(residual < delta, 0.5 * tf.square(residual), delta * residual - 0.5 * tf.square(delta))

    def click_weighted_log_ratio_loss(
            self, output, labels, propensity_weights, name=None):
        """Computes pointwise MSE loss with propensity weighting.

        Args:
            output: (tf.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (tf.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
            relevant example.
            propensity_weights: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.
            name: A string used as the name for this variable scope.

        Returns:
            (tf.Tensor) A single value tensor containing the loss.
        """
        loss = None
        with tf.name_scope(name, "click_weighted_log_ratio_loss", [output]):

            click_predict = output / (propensity_weights + 1e-7)
            click_max = tf.reduce_max(click_predict, axis=1, keep_dims=True)
            click_predict = click_predict / (click_max + 1e-7)

            zero_weight = tf.reduce_mean(labels, axis=1, keep_dims=True)
            one_weight = tf.ones_like(zero_weight) / (zero_weight)
            # # weight = labels / ( + 1e-3)
            # loss = tf.losses.log_loss(labels, click_predict, )
        return loss
    
    def pairwise_debias_propensity_loss(self, output, labels, propensity_weights, name=None):
        output_plus, output_minus = output
        return tf.losses.mean_squared_error(
            tf.concat(self.t_plus_loss_list, axis=1) / self.t_plus_loss_list[0],
            output_plus
        ) + tf.losses.mean_squared_error(
            tf.concat(self.t_minus_loss_list, axis=1) / self.t_minus_loss_list[0],
            output_minus
        )
        

    def pairwise_debias_ranking_loss(self, output, labels, propensity_weights, name=None):
        """Computes pointwise MSE loss with propensity weighting.

        Args:
            output: (tf.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (tf.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
            relevant example.
            propensity_weights: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.
            name: A string used as the name for this variable scope.

        Returns:
            (tf.Tensor) A single value tensor containing the loss.
        """
        t_plus = tf.unstack(propensity_weights[0], axis=1)
        t_minus = tf.unstack(propensity_weights[1], axis=1)

        self.t_plus_loss_list = [None] * self.selection_bias_cutoff
        self.t_minus_loss_list = [None] * self.selection_bias_cutoff
        loss = None
        sliced_output = tf.unstack(output, axis=1)
        sliced_label = tf.unstack(labels, axis=1)
        # delta NDCG
        inverse_idcg = tf.reshape(ultra.utils.inverse_max_dcg(labels), (-1, )) # (B, )
        for i in range(len(sliced_output)):
            for j in range(i + 1, len(sliced_output)):

                s1 = sliced_output[i] # (B, )
                s2 = sliced_output[j] # (B, )
                s12 = tf.sigmoid(s1 - s2) # (B, ), 0~1
                click1 = sliced_label[i] # (B, )
                click2 = sliced_label[j] # (B, )
                cur_labels = click1 - click2 # (B, ), 0/1/-1
                weights = tf.abs(cur_labels) # (B, ), 0/1
                cur_labels = (cur_labels + 1) / 2 # (B, ), 0/1

                pos_loginv = 1.0 / math.log2(i + 2)
                neg_loginv = 1.0 / math.log2(j + 2)
                original = click1 * pos_loginv + click2 * neg_loginv
                changed = click2 * pos_loginv + click1 * neg_loginv
                delta = tf.abs((original - changed) * inverse_idcg) # (B, )

                valid_pair_mask = tf.math.minimum(
                    tf.ones_like(
                        self.labels[i]), tf.nn.relu(
                        self.labels[i] - self.labels[j]))
                pair_loss = valid_pair_mask * delta * (
                    self.pairwise_cross_entropy_loss(
                        s1, s2)
                )
                if self.t_plus_loss_list[i] is None:
                    self.t_plus_loss_list[i] = pair_loss / t_minus[j]
                else: 
                    self.t_plus_loss_list[i] += pair_loss / t_minus[j]
                
                if self.t_plus_loss_list[j] is None:
                    self.t_minus_loss_list[j] = pair_loss / t_plus[i]
                else: 
                    self.t_minus_loss_list[j] += pair_loss / t_plus[i]
                
                if loss is None:
                    loss = pair_loss / \
                        t_plus[i] / t_minus[j]
                else:
                    loss += pair_loss / \
                        t_plus[i] / t_minus[j]

        return tf.reduce_sum(loss) / tf.shape(output)[0]