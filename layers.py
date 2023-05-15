import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
ks = tf.keras

class LParameterAB(GraphBaseLayer):
    def __init__(self, **kwargs):
        super(LParameterAB, self).__init__(**kwargs)
        self.gamma_ab_table = self.add_weight(shape=(95, 95), name="bond_L_ab", trainable=True, dtype="float32",
                                              initializer=tf.initializers.RandomUniform(0.03, 2.0))

        # self.set_weights([np.arange(95*95).reshape(95,95)])

    def call(self, inputs, **kwargs):
        """GetGammaAB for atom number a, b

        Args:
            inputs (tf.RaggedTensor): Atom numbers of a bond. Shape is (batch, None, 2).

        Returns:
            tf.RaggedTensor: Gamma Values for each bond. Shape is (batch, None, 1)
        """
        assert isinstance(inputs, tf.RaggedTensor)
        assert inputs.ragged_rank == 1
        num_vals = inputs.values
        num_vals = tf.sort(num_vals, axis=-1)  # Need to sort since we simply keep with lower triangle part of matrix.
        num = tf.gather(tf.gather(self.gamma_ab_table, num_vals[:, 1]), num_vals[:, 0], axis=-1, batch_dims=1)
        num = tf.expand_dims(num, axis=-1)
        return tf.RaggedTensor.from_row_splits(num, inputs.row_splits, validate=self.ragged_validate)


class ErepForLab(GraphBaseLayer):
    def __init__(self, add_eps: bool = False, **kwargs):
        super(ErepForLab, self).__init__(**kwargs)
        # self.activ_leak = tf.keras.layers.LeakyReLU(alpha=0.01)
        self.add_eps = add_eps

    def call(self, inputs, **kwargs):
        """GetGammaAB for atom number a, b

        Args:
            inputs: [charge_ab, distance, L_ab]

                - charge_ab (tf.RaggedTensor): Charges of atoms for bond. Shape is (batch, None, 2).
                - distance (tf.RaggedTensor): Distance for bond. Shape is (batch, None, 1).
                - L_ab (tf.RaggedTensor): Lab for bond. Shape is (batch, None, 1).

        Returns:
            tf.RaggedTensor: Gamma Values for each bond. Shape is (batch, None, 1)
        """
        assert all([isinstance(x, tf.RaggedTensor) for x in inputs])
        assert all([x.ragged_rank == 1 for x in inputs])
        dist_values = inputs[1].values
        charge_values, l_values = tf.cast(inputs[0].values, dtype=dist_values.dtype), \
                                  tf.cast(inputs[2].values, dtype=dist_values.dtype)
        if self.add_eps:
            l_square = tf.square(l_values + ks.backend.epsilon())
        else:
            l_square = tf.square(l_values)
        yab = tf.math.divide(1.0, tf.sqrt(tf.square(dist_values) + l_square/4.0))
        e_rep = 0.5 * charge_values[:, :1] * charge_values[:, 1:] * (tf.divide(1.0, dist_values) - yab)
        return tf.RaggedTensor.from_row_splits(e_rep, inputs[1].row_splits, validate=self.ragged_validate)
    
    
class ErepForUaUb(GraphBaseLayer):
    def __init__(self, add_eps: bool = False, **kwargs):
        super(ErepForUaUb, self).__init__(**kwargs)
        # self.activ_leak = tf.keras.layers.LeakyReLU(alpha=0.01)
        self.add_eps = add_eps

    def call(self, inputs, **kwargs):
        """GetGammaAB for atom number a, b

        Args:
            inputs: [charge_ab, distance, L_ab]

                - charge_ab (tf.RaggedTensor): Charges of atoms for bond. Shape is (batch, None, 2).
                - distance (tf.RaggedTensor): Distance for bond. Shape is (batch, None, 1).
                - L_ab (tf.RaggedTensor): Lab for bond. Shape is (batch, None, 1).

        Returns:
            tf.RaggedTensor: Gamma Values for each bond. Shape is (batch, None, 1)
        """
        assert all([isinstance(x, tf.RaggedTensor) for x in inputs])
        assert all([x.ragged_rank == 1 for x in inputs])
        dist_values = inputs[1].values
        charge_values, u_values = tf.cast(inputs[0].values, dtype=dist_values.dtype), \
                                  tf.cast(inputs[2].values, dtype=dist_values.dtype)
        #if self.add_eps:
        #    u_square = tf.square(u_values[:, :1] + ks.backend.epsilon() + tf.math.divide_no_nan(1.0, u_values[:, 1:] + ks.backend.epsilon()))
        #else:
        #    u_square = tf.square(u_values[:, :1]) + tf.math.divide_no_nan(1.0, u_values[:, 1:]))
        
        u_square = tf.square(u_values[:, :1])
        yab = tf.math.divide(1.0,
            tf.sqrt(tf.square(dist_values) + u_square/4.0)
            )
        e_rep = 0.5 * charge_values[:, :1] * charge_values[:, 1:] * (tf.divide(1.0, dist_values) - yab)
        return tf.RaggedTensor.from_row_splits(e_rep, inputs[1].row_splits, validate=self.ragged_validate)
