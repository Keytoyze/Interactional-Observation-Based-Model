import tensorflow as tf

from ultra.propensity_model.base_propensity_model import BasePropensityModel


class ClickData(BasePropensityModel):

    def __init__(self, hparams_str, **kwargs):
        print("Propensity: use click data")

    def build(self, is_training, click_label, learning_model):
        epsilon = tf.get_variable("epsilon", shape=[1])
        epsilon = tf.clip_by_value(epsilon, 1e-7, 1e-6)
        return tf.ones_like(click_label) / 2 + epsilon

