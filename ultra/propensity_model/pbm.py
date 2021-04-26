import tensorflow as tf
import ultra
from ultra.propensity_model.base_propensity_model import BasePropensityModel


class PBM(BasePropensityModel):

    def __init__(self, hparams_str, **kwargs):
        print("Propensity: use PBM-IPS")

        self.hparams = ultra.utils.hparams.HParams(
            constant_propensity_initialization=True
        )
        self.hparams.parse(hparams_str)
        print("PBM params: " + self.hparams.to_json())

    def build(self, is_training, click_label, learning_model):
        list_size = click_label.shape[1]
        batch_size = tf.shape(click_label)[0]
        initializer = tf.initializers.constant(0.001) if self.hparams.constant_propensity_initialization else None
        weight = tf.get_variable("pbm_weight", (1, list_size), initializer=initializer)
        propensity = tf.tile(weight, [batch_size, 1])
        return self.to_prob(propensity)
