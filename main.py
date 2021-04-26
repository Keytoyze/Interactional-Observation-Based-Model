"""Training and testing unbiased learning to rank algorithms.

See the following paper for more information about different algorithms.
    
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
import json
import ultra

#rank list size should be read from data
tf.app.flags.DEFINE_string("data_dir", "./tests/data/", "The directory of the experimental dataset.")
tf.app.flags.DEFINE_string("train_data_prefix", "train", "The name prefix of the training data in data_dir.")
tf.app.flags.DEFINE_string("valid_data_prefix", "valid", "The name prefix of the validation data in data_dir.")
tf.app.flags.DEFINE_string("test_data_prefix", "test", "The name prefix of the test data in data_dir.")
tf.app.flags.DEFINE_string("model_dir", "./tests/tmp_model/", "The directory for model and intermediate outputs.")
tf.app.flags.DEFINE_string("output_dir", "./tests/tmp_output/", "The directory to output results.")

# model 
tf.app.flags.DEFINE_string("setting_file", "./example/offline_setting/dla_exp_settings.json", "A json file that contains all the settings of the algorithm.")

# general training parameters
tf.app.flags.DEFINE_integer("batch_size", 512,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("max_list_cutoff", 0,
                            "The maximum number of top documents to consider in each rank list (0: no limit).")
tf.app.flags.DEFINE_integer("selection_bias_cutoff", 10,
                            "The maximum number of top documents to be shown to user (which creates selection bias) in each rank list (0: no limit).")
tf.app.flags.DEFINE_integer("max_train_iteration", 10000,
                            "Limit on the iterations of training (0: no limit).")
tf.app.flags.DEFINE_integer("start_saving_iteration", 0,
                            "The minimum number of iterations before starting to test and save models. (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 50,
                            "How many training steps to do per checkpoint.")

tf.app.flags.DEFINE_boolean("test_while_train", False,
                            "Set to True to test models during the training process.")
tf.app.flags.DEFINE_boolean("test_only", False,
                            "Set to True for testing models only.")

tf.app.flags.DEFINE_string("gpu", "",
                           "Set the CUDA visible GPU devices.")
tf.app.flags.DEFINE_integer("early_stop_step_patient", 15000,
                            "Stop training after no improve")
tf.app.flags.DEFINE_integer("reduce_propensity_lr_step", 500000,
                            "Reduce propensity lr after no improve on KL")
tf.app.flags.DEFINE_integer("reduce_lr_step", 100000,
                            "Reduce propensity lr after no improve on objective metric")
tf.app.flags.DEFINE_string("validation_steps", "[]", 
                            "Validation steps. Default: 0:max_train_iteration:steps_per_checkpoint")

FLAGS = tf.app.flags.FLAGS

check_validation = False

def create_model(session, exp_settings, data_set, forward_only):
    """Create model and initialize or load parameters in session.
    
        Args:
            session: (tf.Session) The session used to run tensorflow models
            exp_settings: (dictionary) The dictionary containing the model settings.
            data_set: (Raw_data) The dataset used to build the input layer.
            forward_only: Set true to conduct prediction only, false to conduct training.
    """

    model = ultra.utils.find_class(exp_settings['learning_algorithm'])(data_set, exp_settings, forward_only)

    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt:
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model

# Validate model
def validate_model(data_set, data_input_feed, model, sess):
    it = 0
    count_batch = 0.0
    summary_list = []
    batch_size_list = []
    while it < len(data_set.initial_list):
        input_feed, info_map = data_input_feed.get_next_batch(it, data_set, check_validation=check_validation)
        _, _, summary = model.step(sess, input_feed, True)
        summary_list.append(summary)
        batch_size_list.append(len(info_map['input_list']))
        it += batch_size_list[-1]
        count_batch += 1.0
    return ultra.utils.merge_TFSummary(summary_list, batch_size_list)

def train(exp_settings):
    # Prepare data.
    print("Reading data in %s" % FLAGS.data_dir)
    train_set = ultra.utils.read_data(FLAGS.data_dir, FLAGS.train_data_prefix, FLAGS.max_list_cutoff)
    ultra.utils.find_class(exp_settings['train_input_feed']).preprocess_data(train_set, exp_settings['train_input_hparams'], exp_settings)
    valid_set = ultra.utils.read_data(FLAGS.data_dir, FLAGS.valid_data_prefix, FLAGS.max_list_cutoff)
    ultra.utils.find_class(exp_settings['train_input_feed']).preprocess_data(valid_set, exp_settings['train_input_hparams'], exp_settings)

    print("Train Rank list size %d" % train_set.rank_list_size)
    print("Valid Rank list size %d" % valid_set.rank_list_size)
    exp_settings['max_candidate_num'] = max(train_set.rank_list_size, valid_set.rank_list_size)

    test_set = ultra.utils.read_data(FLAGS.data_dir, FLAGS.test_data_prefix, FLAGS.max_list_cutoff)
    ultra.utils.find_class(exp_settings['train_input_feed']).preprocess_data(test_set, exp_settings['train_input_hparams'], exp_settings)
    print("Test Rank list size %d" % test_set.rank_list_size)
    exp_settings['max_candidate_num'] = max(test_set.rank_list_size, exp_settings['max_candidate_num'])
    test_set.pad(exp_settings['max_candidate_num'])

    if 'selection_bias_cutoff' not in exp_settings: # check if there is a limit on the number of items per training query.
        exp_settings['selection_bias_cutoff'] = FLAGS.selection_bias_cutoff if FLAGS.selection_bias_cutoff > 0 else exp_settings['max_candidate_num']

    exp_settings['selection_bias_cutoff'] = min(exp_settings['selection_bias_cutoff'], exp_settings['max_candidate_num'])
    print('Users can only see the top %d documents for each query in training.' % exp_settings['selection_bias_cutoff'])

    # Pad data
    train_set.pad(exp_settings['max_candidate_num'])
    valid_set.pad(exp_settings['max_candidate_num'])

    validation_steps = eval(FLAGS.validation_steps)
    if len(validation_steps) == 0:
        validation_steps = [FLAGS.steps_per_checkpoint * i for i in range(FLAGS.max_train_iteration // FLAGS.steps_per_checkpoint)]
    print("validation step: ", validation_steps)

    ndcg_10_records = []
    kl_records = []

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Create model based on the input layer.
        print("Creating model...")
        model = create_model(sess, exp_settings, train_set, False)
        #model.print_info()

        # Create data feed
        train_input_feed = ultra.utils.find_class(exp_settings['train_input_feed'])(model, FLAGS.batch_size, exp_settings['train_input_hparams'], sess)
        valid_input_feed = ultra.utils.find_class(exp_settings['valid_input_feed'])(model, 1024, exp_settings['valid_input_hparams'], sess)
        test_input_feed = None
        test_input_feed = ultra.utils.find_class(exp_settings['test_input_feed'])(model, 1024, exp_settings['test_input_hparams'], sess)

        # Create tensorboard summarizations.
        train_writer = tf.summary.FileWriter(FLAGS.model_dir + '/train_log',
                                             sess.graph)
        valid_writer = tf.summary.FileWriter(FLAGS.model_dir + '/valid_log')
        test_writer = tf.summary.FileWriter(FLAGS.model_dir + '/test_log')

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        last_best_step = 0
        last_best_step_for_reduce = 0
        last_best_propensity_step = 0
        freeze_step = 0
        previous_losses = []
        best_perf = None
        best_propensity_kl = None

        valid_log = {}

        while True:
            # Get a batch and make a step.
            start_time = time.time()
            input_feed, info_map = train_input_feed.get_batch(train_set, check_validation=check_validation)
            step_loss, _, summary = model.step(sess, input_feed, False)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1
            train_writer.add_summary(summary, model.global_step.eval())

            # if current_step % FLAGS.reduce_lr_step == 0:
            #     reduce_op = tf.assign(model.learning_rate, model.learning_rate / 2)
            #     sess.run(reduce_op)

                # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                print ("global step %d batch-size %d learning rate %.6f propensity lr %.6lf step-time %.2f loss "
                       "%.4f" % (model.global_step.eval(), len(info_map['input_list']), model.learning_rate.eval(),
                                 model.propensity_learning_rate.eval(),
                                 step_time, loss))
                previous_losses.append(loss)

                if current_step in validation_steps:
                    valid_summary = validate_model(valid_set, valid_input_feed, model, sess)
                    valid_writer.add_summary(valid_summary, model.global_step.eval())
                    print("  valid: %s" % (
                        ' '.join(['%s:%.3f' % (x.tag, x.simple_value) for x in valid_summary.value])
                    ))
                    valid_log[current_step] = dict((x.tag, x.simple_value) for x in valid_summary.value)
                    valid_log[current_step]['step'] = current_step
                    valid_log[current_step]['freeze_step'] = freeze_step
                    ndcg_10_records.append(valid_log[current_step]['ndcg_10'])
                    if 'KL' in valid_log[current_step]:
                        kl_records.append(valid_log[current_step]['KL'])
                    with open(os.path.join(FLAGS.model_dir, "valid.json"), "w") as f:
                        json.dump(valid_log, f)

                    # Save checkpoint if the objective metric on the validation set is better
                    if "objective_metric" in exp_settings:
                        for x in valid_summary.value:
                            if x.tag == exp_settings["objective_metric"]:
                                if current_step >= FLAGS.start_saving_iteration:
                                    if best_perf == None or best_perf < x.simple_value:
                                        checkpoint_path = os.path.join(FLAGS.model_dir, "%s.ckpt" % exp_settings['learning_algorithm'])
                                        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                                        best_perf = x.simple_value
                                        print('Save model, valid %s:%.3f' % (x.tag, best_perf))
                                        last_best_step = current_step
                                        # last_best_step_for_reduce = current_step

                                        with open(os.path.join(FLAGS.model_dir, "valid.json"), "w") as f:
                                            json.dump(valid_log, f)

                                        if FLAGS.test_while_train:
                                            test_summary = validate_model(test_set, test_input_feed, model, sess)
                                            test_writer.add_summary(test_summary, model.global_step.eval())
                                            print("  test: %s" % (
                                                ' '.join(['%s:%.3f' % (x.tag, x.simple_value) for x in test_summary.value])
                                            ))

                                            test_log = dict((x.tag, x.simple_value) for x in test_summary.value)
                                            test_log['step'] = current_step
                                            test_log['freeze_step'] = freeze_step
                                            test_log['ndcg_10_records'] = ndcg_10_records
                                            test_log['kl_records'] = kl_records
                                            with open(os.path.join(FLAGS.model_dir, "test.json"), "w") as f:
                                                json.dump(test_log, f)
                                        break
                    # Save checkpoint if there is no objective metic
                    if best_perf == None and current_step > FLAGS.start_saving_iteration:
                        checkpoint_path = os.path.join(FLAGS.model_dir, "%s.ckpt" % exp_settings['learning_algorithm'])
                        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                    if loss == float('inf'):
                        break

                    current_kl = None
                    for x in valid_summary.value:
                        if x.tag != 'KL':
                            continue
                        current_kl = x.simple_value
                        if best_propensity_kl == None or best_propensity_kl >= x.simple_value:
                            best_propensity_kl = current_kl
                            # last_best_propensity_step = current_step

                    if current_step - last_best_propensity_step > FLAGS.reduce_propensity_lr_step:
                        print("Reduce propensity LR due to last best KL step is: %d (%.5lf)" %
                            (last_best_propensity_step, best_propensity_kl))
                        last_best_propensity_step = current_step
                        best_propensity_kl = current_kl
                        reduce_propensity = tf.assign(model.propensity_learning_rate, model.propensity_learning_rate / 2)
                        sess.run(reduce_propensity)

                    if current_step - last_best_step_for_reduce > FLAGS.reduce_lr_step:
                        print("Reduce LR due to last best step is: %d (%.4lf)" %
                            (last_best_step_for_reduce, best_perf))
                        last_best_step_for_reduce = current_step
                        reduce_propensity = tf.assign(model.learning_rate, model.learning_rate / 2)
                        sess.run(reduce_propensity)

                step_time, loss = 0.0, 0.0
                sys.stdout.flush()

                if FLAGS.max_train_iteration > 0 and current_step - freeze_step > FLAGS.max_train_iteration:
                    break

                if current_step - last_best_step > FLAGS.early_stop_step_patient:
                    break
    tf.reset_default_graph()
    with tf.Session(config=config) as sess:
        # test
        model = create_model(sess, exp_settings, train_set, False)
        test_summary = validate_model(test_set, test_input_feed, model, sess)
        test_writer.add_summary(test_summary, model.global_step.eval())
        print("  test: %s" % (
            ' '.join(['%s:%.3f' % (x.tag, x.simple_value) for x in test_summary.value])
        ))

        test_log = dict((x.tag, x.simple_value) for x in test_summary.value)
        test_log['step'] = last_best_step
        test_log['freeze_step'] = freeze_step
        test_log['ndcg_10_records'] = ndcg_10_records
        test_log['kl_records'] = kl_records
        with open(os.path.join(FLAGS.model_dir, "test.json"), "w") as f:
            json.dump(test_log, f)



def test(exp_settings):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Load test data.
        print("Reading data in %s" % FLAGS.data_dir)
        test_set = ultra.utils.read_data(FLAGS.data_dir, FLAGS.test_data_prefix, FLAGS.max_list_cutoff)
        ultra.utils.find_class(exp_settings['train_input_feed']).preprocess_data(test_set, exp_settings['train_input_hparams'], exp_settings)
        exp_settings['max_candidate_num'] = test_set.rank_list_size
        if 'selection_bias_cutoff' not in exp_settings: # check if there is a limit on the number of items per training query.
            exp_settings['selection_bias_cutoff'] = FLAGS.selection_bias_cutoff if FLAGS.selection_bias_cutoff > 0 else exp_settings['max_candidate_num']

        exp_settings['selection_bias_cutoff'] = min(exp_settings['selection_bias_cutoff'], exp_settings['max_candidate_num'])

        test_set.pad(exp_settings['max_candidate_num'])

        # Create model and load parameters.
        model = create_model(sess, exp_settings, test_set, True)

        # Create input feed
        test_input_feed = ultra.utils.find_class(exp_settings['test_input_feed'])(model, FLAGS.batch_size, exp_settings['test_input_hparams'], sess)

        test_writer = tf.summary.FileWriter(FLAGS.model_dir + '/test_log')

        rerank_scores = []
        summary_list = []
        # Start testing.

        it = 0
        count_batch = 0.0
        batch_size_list = []
        while it < len(test_set.initial_list):
            input_feed, info_map = test_input_feed.get_next_batch(it, test_set, check_validation=check_validation)
            _, output_logits, summary = model.step(sess, input_feed, True)
            summary_list.append(summary)
            batch_size_list.append(len(info_map['input_list']))
            for x in range(batch_size_list[-1]):
                rerank_scores.append(output_logits[x])
            it += batch_size_list[-1]
            count_batch += 1.0
            print("Testing {:.0%} finished".format(float(it)/len(test_set.initial_list)), end="\r", flush=True)

        print("\n[Done]")
        test_summary = ultra.utils.merge_TFSummary(summary_list, batch_size_list)
        test_writer.add_summary(test_summary, it)
        print("  eval: %s" % (
            ' '.join(['%s:%.3f' % (x.tag, x.simple_value) for x in test_summary.value])
        ))

        #get rerank indexes with new scores
        rerank_lists = []
        for i in range(len(rerank_scores)):
            scores = rerank_scores[i]
            rerank_lists.append(sorted(range(len(scores)), key=lambda k: scores[k], reverse=True))

        if not os.path.exists(FLAGS.output_dir):
            os.makedirs(FLAGS.output_dir)
        ultra.utils.output_ranklist(test_set, rerank_scores, FLAGS.output_dir, FLAGS.test_data_prefix)


    return

def main(_):
    if FLAGS.gpu != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    exp_settings = json.load(open(FLAGS.setting_file))
    if FLAGS.test_only:
        test(exp_settings)
    else:
        train(exp_settings)

if __name__ == "__main__":
    tf.app.run()
