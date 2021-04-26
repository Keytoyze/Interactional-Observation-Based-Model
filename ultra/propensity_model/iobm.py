import tensorflow as tf
import ultra
from ultra.propensity_model.base_propensity_model import BasePropensityModel


class IOBM(BasePropensityModel):

    def __init__(self, hparams_str, **kwargs):
        print("Propensity: use IOBM")

        self.hparams = ultra.utils.hparams.HParams(
            activation="tanh",
            units=8,
            embedding_size=4,
            position_embedding_size=4,
            bidirection=True
        )
        self.hparams.parse(hparams_str)
        print("IOBM params: " + self.hparams.to_json())

    def lstm_layer(self, inpt):
        x = tf.keras.layers.LSTM(units=self.hparams.units, activation=self.hparams.activation,
                                 return_sequences=True)(inpt)
        return x

    # context: (B, T, C1)
    # sequence: (B, T, C2)
    # return: attention score, output
    def additive_attention(self, context, sequence):
        x = tf.concat([context, sequence], axis=-1) # (B, T, C1 + C2)
        C2 = sequence.shape[-1]
        x = tf.keras.layers.Dense(C2, activation='tanh')(x) # (B, T, C2)
        x = tf.nn.softmax(x) # (B, T, C2)
        x = x * tf.cast(C2, tf.float32)
        print("additive_attention: %s vs %s => %s" % (context.shape, sequence.shape, x.shape))
        return x, x * sequence

    def build(self, is_training, click_label, learning_model):

        list_size = click_label.shape.as_list()[1]
        batch_size = tf.shape(click_label)[0]
        inputs = []

        # context
        context = tf.stop_gradient(tf.add_n(learning_model.context_embedding) / tf.cast(list_size, tf.float32))
        # context = context[:, :700]
        context_dim = context.shape[-1]
        print("context dim: " + str(context_dim))
        context = tf.reshape(context, (batch_size, 1, context_dim))
        context = tf.tile(context, [1, list_size, 1])

        # position embedding
        position = tf.reshape(tf.range(list_size), (1, -1, 1))
        position = tf.keras.layers.Embedding(list_size, self.hparams.position_embedding_size)(position)
        position = tf.squeeze(position, axis=[2])
        position = tf.tile(position, [batch_size, 1, 1])
        inputs.append(position)

        # click label embedding
        click_label = tf.expand_dims(click_label, axis=-1)
        if self.hparams.embedding_size != 0:
            click_label = tf.keras.layers.Embedding(2, self.hparams.embedding_size)(click_label)
            click_label = tf.squeeze(click_label, axis=[2])
        else:
            click_label = click_label * 2 - 1
        inputs.append(click_label)
        
        x = tf.concat(inputs, axis=-1)

        # attention
        att_x, x = self.additive_attention(context, x)
        l1 = self.hparams.position_embedding_size
        l2 = l1 + self.hparams.embedding_size
        l3 = l2 + 1
        tf.summary.scalar('pos_att', tf.reduce_mean(att_x[:, :, :l1]), collections=['eval'])
        tf.summary.scalar('clk_att', tf.reduce_mean(att_x[:, :, l1:l2]), collections=['eval'])
    
        # shift
        x = tf.unstack(x, axis=1)
        x.insert(0, tf.zeros_like(x[0]))
        x.append(tf.zeros_like(x[0]))

        forward = tf.stack(x[:-2], axis=1)
        p = self.lstm_layer(forward)

        if self.hparams.bidirection:
            backward = tf.stack(x[-1:1:-1], axis=1)
            q = self.lstm_layer(backward)
            q = tf.reverse(q, axis=[1])
            x = tf.concat([p, q], axis=-1)
            att_y, x = self.additive_attention(context, x)
            y = tf.keras.layers.Dense(1)(x)
            y = tf.squeeze(y, axis=[-1])
            y2 = tf.keras.layers.Dense(1)(x)
            y2 = tf.squeeze(y2, axis=[-1])

            l1 = self.hparams.units
            l2 = l1 + self.hparams.units
            tf.summary.scalar('forward_att', tf.reduce_mean(att_y[:, :, :l1]), collections=['eval'])
            tf.summary.scalar('backward_att', tf.reduce_mean(att_y[:, :, l1:l2]), collections=['eval'])

        else:
            y = tf.keras.layers.Dense(1)(p)
            y = tf.squeeze(y, axis=[-1])

        y = self.to_prob(y)
        return y
