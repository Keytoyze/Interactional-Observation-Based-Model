import tensorflow as tf

from ultra.propensity_model.base_propensity_model import BasePropensityModel


class DCM(BasePropensityModel):

    def __init__(self, hparams_str):

        print("Propensity: use DCM-IPS")

    def build(self, is_training, click_label, learning_model):
        list_size = click_label.shape[1]

        x = tf.unstack(click_label, axis=1)
        x.insert(0, tf.ones_like(x[0]))
        x.pop()
        x = tf.stack(x, axis=1)

        dcm_weights = (tf.tanh(tf.get_variable("dcm_weights", shape=(1, list_size))) + 1) / 2

        x = 1 - dcm_weights * x
        x = tf.cumprod(x, axis=1)
        return x
