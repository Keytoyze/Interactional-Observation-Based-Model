import tensorflow as tf
import ultra
from ultra.propensity_model.base_propensity_model import BasePropensityModel


class CPBM(BasePropensityModel):

    def __init__(self, hparams_str, **kwargs):
        print("Propensity: use CPBM-IPS")

        self.hparams = ultra.utils.hparams.HParams(
            constant_propensity_initialization=True
        )
        self.hparams.parse(hparams_str)
        print("CPBM params: " + self.hparams.to_json())

    def build(self, is_training, click_label, learning_model):
        list_size = click_label.shape[1]

        context = tf.stop_gradient(tf.add_n(learning_model.context_embedding) / tf.cast(list_size, tf.float32))
        print(context.shape)
        # context = context[:, :700]
        print("context dim: " + str(context.shape[-1]))

        x = tf.keras.layers.Dense(256, activation="elu")(context)
        x = tf.keras.layers.Dense(list_size)(x)

        return self.to_prob(x)
