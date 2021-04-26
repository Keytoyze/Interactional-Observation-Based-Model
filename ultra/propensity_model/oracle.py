import tensorflow as tf

from ultra.propensity_model.base_propensity_model import BasePropensityModel


class Oracle(BasePropensityModel):

    def __init__(self, hparams_str, **kwargs):
        print("Propensity: use oracle-ips")

    def anti_sigmoid(self, x):
        no_use = tf.get_variable("no_use", shape=[1])
        maxx = tf.reduce_max(x, axis=1, keepdims=True)
        minx = tf.reduce_min(x, axis=1, keepdims=True)
        x = (x - minx) / (maxx - minx + 1e-7) - 0.5  # [-1/2, 1/2]
        return (tf.atanh(x) * (no_use * 0 + 1))

    def build(self, is_training, click_label, learning_model):
        list_size = click_label.shape[1]
        epsilon = tf.get_variable("epsilon", shape=[1])
        epsilon = tf.clip_by_value(epsilon, 1e-7, 1e-7)
        return (tf.transpose(tf.stack([learning_model.exam_p_list[0]]))) / (
                    tf.transpose(tf.stack(learning_model.exam_p_list[:list_size])) + epsilon)
