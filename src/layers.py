import tensorflow as tf

# NOTE: BOTH of these flatten because this is the only way to ENSURE consistent sizes for the input and output of the model

class FlattenHistogram2D(tf.keras.layers.Layer):
    def __init__(self):
        super(FlattenHistogram2D, self).__init__()

    def call(self, inputs):
        # make three copies of the input
        # one for each dimension
        # then stack them together
        # and return the result
        i1, i2, i3 = tf.identity(inputs), tf.identity(inputs), tf.identity(inputs)

        # flatten the inputs by summing along each dimension
        # then reshape them to be the same shape as the input
        # but with two dimensions
        i1 = tf.reshape(tf.reduce_sum(i1, axis=0), (-1, 1))
        i2 = tf.reshape(tf.reduce_sum(i2, axis=1), (-1, 1))
        i3 = tf.reshape(tf.reduce_sum(i3, axis=2), (-1, 1))


        # stack the inputs together
        # and return the result
        return tf.concat([i1, i2, i3], axis=-1)


class FlattenHistogram1D(tf.keras.layers.Layer):
    def __init__(self):
        super(FlattenHistogram1D, self).__init__()

    def call(self, inputs):
        # make three copies of the input
        # one for each dimension
        # then stack them together
        # and return the result
        i1, i2, i3 = tf.identity(inputs), tf.identity(inputs), tf.identity(inputs)

        # flatten the inputs by summing along each dimension
        # then reshape them to be the same shape as the input
        # but with only one dimension
        i1 = tf.reshape(tf.reduce_sum(tf.reduce_sum(i1, axis=0), axis=0), (-1, 1))
        i2 = tf.reshape(tf.reduce_sum(tf.reduce_sum(i2, axis=1), axis=1), (-1, 1))
        i3 = tf.reshape(tf.reduce_sum(tf.reduce_sum(i3, axis=0), axis=1), (-1, 1))

        # stack the inputs together
        # and return the result
        return tf.concat([i1, i2, i3], axis=-1)