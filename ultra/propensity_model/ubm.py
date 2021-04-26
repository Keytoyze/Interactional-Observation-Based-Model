import tensorflow as tf
import ultra
from ultra.propensity_model.base_propensity_model import BasePropensityModel


class UBM(BasePropensityModel):

    def __init__(self, hparams_str, **kwargs):
        print("Propensity: use UBM-IPS")

        self.hparams = ultra.utils.hparams.HParams(
            constant_propensity_initialization=True
        )
        self.hparams.parse(hparams_str)
        print("UBM params: " + self.hparams.to_json())
    
    def build_one_result(self, click_label, pre_fix):
        # click_label = tf.zeros_like(click_label)
        list_size = click_label.shape[1]
        batch_size = tf.shape(click_label)[0]
        no_click_label = tf.unstack(1 - tf.cast(click_label, dtype=tf.int32), axis=1)
        propensities = []

        weight = []  # (rank, distance) => weight
        for i in range(list_size):
            weight.append(tf.get_variable(pre_fix + "ubm_weight_%d" % i, shape=(i + 1)))
        distance = tf.zeros((batch_size,), dtype=tf.int32)

        for i in range(list_size):
            propensities.append(tf.gather(weight[i], distance))
            distance = (distance + 1) * no_click_label[i]

        return tf.stack(propensities, axis=1)


    def build(self, is_training, click_label, learning_model):
            return self.to_prob(self.build_one_result(click_label, ""))
