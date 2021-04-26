import tensorflow as tf

from ultra.propensity_model.base_propensity_model import BasePropensityModel


class LabeledData(BasePropensityModel):

    def __init__(self, hparams_str, **kwargs):
        print("Propensity: use Labeled-Data")

    def build(self, is_training, click_label, learning_model, double_output=False):
        epsilon = tf.get_variable("epsilon", shape=[])
        epsilon = tf.clip_by_value(epsilon, 1e-7, 1e-7)
        return tf.ones_like(click_label) + epsilon