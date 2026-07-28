import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def apply_clahe_rgb(image):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to an RGB image.
    Converts image to LAB color space, applies CLAHE on Lightness channel, and converts back.
    """
    # Ensure image is in uint8 format (0-255)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
        
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    merged = cv2.merge((cl, a_channel, b_channel))
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return result.astype(np.float32) / 255.0

def load_and_preprocess_image(file_path, target_size=(224, 224), use_clahe=True):
    """
    Loads an image from path, resizes it, normalizes it, and optionally applies CLAHE.
    """
    # Read image using OpenCV (convert BGR to RGB)
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Could not read image from path: {file_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Apply CLAHE
    if use_clahe:
        img_normalized = apply_clahe_rgb(img)
    else:
        img_normalized = img.astype(np.float32) / 255.0
        
    return img_normalized

def get_data_generators(train_dir, val_dir, test_dir, batch_size=32, target_size=(224, 224)):
    """
    Creates and returns Keras ImageDataGenerators for train, validation, and test datasets.
    Custom preprocessing function is used to apply CLAHE automatically.
    """
    # Custom preprocessing to integrate CLAHE into the generator flow
    def custom_preprocessing(image):
        # Image passed by ImageDataGenerator is a float32 array in [0, 255] or already normalized
        return apply_clahe_rgb(image)

    train_datagen = ImageDataGenerator(
        preprocessing_function=custom_preprocessing,
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    # Validation and test generators only apply CLAHE, no augmentations
    val_test_datagen = ImageDataGenerator(
        preprocessing_function=custom_preprocessing
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator, test_generator
