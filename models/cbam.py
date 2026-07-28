import tensorflow as tf
from tensorflow.keras import layers

class ChannelAttention(layers.Layer):
    def __init__(self, reduction_ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_layer_one = layers.Dense(
            channel // self.reduction_ratio,
            activation='relu',
            kernel_initializer='he_normal',
            use_bias=True,
            bias_initializer='zeros'
        )
        self.shared_layer_two = layers.Dense(
            channel,
            kernel_initializer='he_normal',
            use_bias=True,
            bias_initializer='zeros'
        )
        super(ChannelAttention, self).build(input_shape)

    def call(self, inputs):
        # Average pooling
        avg_pool = layers.GlobalAveragePooling2D()(inputs)
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)

        # Max pooling
        max_pool = layers.GlobalMaxPooling2D()(inputs)
        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)

        # Sum attention maps and apply sigmoid
        added = layers.add([avg_pool, max_pool])
        sigmoid = layers.Activation('sigmoid')(added)
        
        # Reshape to permit broad-casting multiply
        sigmoid = layers.Reshape((1, 1, inputs.shape[-1]))(sigmoid)
        return layers.multiply([inputs, sigmoid])

    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config


class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv2d = layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            strides=1,
            padding='same',
            activation='sigmoid',
            kernel_initializer='he_normal',
            use_bias=False
        )
        super(SpatialAttention, self).build(input_shape)

    def call(self, inputs):
        # Average pool along channel dimension
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        # Max pool along channel dimension
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        
        # Concatenate pool maps
        concat = layers.concatenate([avg_pool, max_pool], axis=-1)
        
        # Convolve to spatial attention map
        attention = self.conv2d(concat)
        return layers.multiply([inputs, attention])

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


def CBAMBlock(inputs, reduction_ratio=8, kernel_size=7):
    """
    Applies CBAM attention (Channel Attention followed by Spatial Attention) on the input tensor.
    """
    x = ChannelAttention(reduction_ratio=reduction_ratio)(inputs)
    x = SpatialAttention(kernel_size=kernel_size)(x)
    return x
