import tensorflow as tf
from tensorflow.keras import layers, Model
from models.backbone import get_backbones
from models.adaptive_fusion import AdaptiveWeightedFusion
from models.cbam import CBAMBlock

def build_adf_net(input_shape=(224, 224, 3), num_classes=3, projection_dim=512):
    """
    Builds the complete ADF-Net architecture:
    1. Dual Backbone Feature Extraction (EfficientNetB0 and DenseNet121)
    2. Conv2D 1x1 Projections to match channel dimensions
    3. Adaptive Weighted Fusion
    4. CBAM Attention Block (Channel + Spatial Attention)
    5. Global Average Pooling
    6. Double Dense Classifier Head with Batch Normalization and Dropout
    """
    # Main Input
    inputs = layers.Input(shape=input_shape)
    
    # Load backbones and extract projected features
    backbone_model = get_backbones(input_shape=input_shape, projection_dim=projection_dim)
    feat_effnet, feat_densenet = backbone_model(inputs)
    
    # Adaptive Feature Fusion
    fused_features = AdaptiveWeightedFusion(name="adaptive_weighted_fusion")([feat_effnet, feat_densenet])
    
    # Attention Block (CBAM)
    refined_features = CBAMBlock(fused_features, reduction_ratio=8, kernel_size=7)
    
    # Global Average Pooling
    pooled = layers.GlobalAveragePooling2D()(refined_features)
    
    # Classification Head
    x = layers.Dense(256, kernel_initializer='he_normal')(pooled)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(128, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Final Softmax Layer
    outputs = layers.Dense(num_classes, activation='softmax', name='prediction_output')(x)
    
    # Complete Compiled Model
    model = Model(inputs=inputs, outputs=outputs, name="ADF_Net")
    return model
