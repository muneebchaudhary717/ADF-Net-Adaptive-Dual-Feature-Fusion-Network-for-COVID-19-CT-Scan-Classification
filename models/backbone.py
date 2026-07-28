import tensorflow as tf
from tensorflow.keras import layers, Model

def get_backbones(input_shape=(224, 224, 3), projection_dim=512):
    """
    Initializes pre-trained backbones (EfficientNet-B0 and DenseNet-121) 
    and adds a 1x1 Convolution projection layer to match feature dimensions.
    """
    inputs = layers.Input(shape=input_shape)
    
    # 1. EfficientNetB0 Backbone
    effnet = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )
    effnet.trainable = False  # Freeze backbone initially
    effnet_features = effnet.output  # Yields (batch, 7, 7, 1280)
    
    # Project EfficientNet features to projection_dim (e.g. 512)
    proj_effnet = layers.Conv2D(
        filters=projection_dim,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer='he_normal',
        name='efficientnet_projection'
    )(effnet_features)
    
    # 2. DenseNet121 Backbone
    densenet = tf.keras.applications.DenseNet121(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )
    densenet.trainable = False  # Freeze backbone initially
    densenet_features = densenet.output  # Yields (batch, 7, 7, 1024)
    
    # Project DenseNet features to projection_dim (e.g. 512)
    proj_densenet = layers.Conv2D(
        filters=projection_dim,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer='he_normal',
        name='densenet_projection'
    )(densenet_features)
    
    # Return a model that accepts the inputs and returns the projected feature maps
    backbone_model = Model(inputs=inputs, outputs=[proj_effnet, proj_densenet], name="adf_backbone")
    return backbone_model
