import tensorflow as tf
from tensorflow.keras import layers

class AdaptiveWeightedFusion(layers.Layer):
    """
    Custom Keras Layer that learns the weights to fuse two input feature maps.
    The weights are normalized using Softmax so that W1 + W2 = 1.
    """
    def __init__(self, **kwargs):
        super(AdaptiveWeightedFusion, self).__init__(**kwargs)

    def build(self, input_shape):
        # We expect a list of two inputs of the same shape
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError("AdaptiveWeightedFusion layer requires a list of exactly 2 inputs.")
            
        if input_shape[0] != input_shape[1]:
            raise ValueError(f"Inputs must have the same shape. Got {input_shape[0]} and {input_shape[1]}")

        # Initialize the raw trainable weights
        self.fusion_weights = self.add_weight(
            name="fusion_weights",
            shape=(2,),
            initializer=tf.keras.initializers.Constant(value=0.5),
            trainable=True,
            dtype=tf.float32
        )
        super(AdaptiveWeightedFusion, self).build(input_shape)

    def call(self, inputs):
        feature_a, feature_b = inputs
        
        # Normalize the weights using Softmax
        normalized_weights = tf.nn.softmax(self.fusion_weights)
        w1 = normalized_weights[0]
        w2 = normalized_weights[1]
        
        # Perform the weighted fusion
        fused = (w1 * feature_a) + (w2 * feature_b)
        return fused

    def get_config(self):
        config = super(AdaptiveWeightedFusion, self).get_config()
        return config
