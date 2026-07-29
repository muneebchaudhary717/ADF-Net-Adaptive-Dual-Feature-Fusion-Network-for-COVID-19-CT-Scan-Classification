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

def get_data_generators(train_dir, val_dir, test_dir, batch_size=32, target_size=(224, 224), dataset_dir=None):
    """
    Creates and returns Keras ImageDataGenerators for train, validation, and test datasets.
    If split folders do not exist, it automatically falls back to dynamically loading and
    splitting the dataset in-memory.
    """
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical
    import os
    import glob

    # Custom preprocessing to integrate CLAHE into the generator flow
    def custom_preprocessing(image):
        return apply_clahe_rgb(image)

    # Check if the standard train directory exists
    if not os.path.exists(train_dir):
        flat_dir = dataset_dir if (dataset_dir and os.path.exists(dataset_dir)) else os.path.dirname(train_dir)
        if not os.path.exists(flat_dir):
            flat_dir = train_dir
            
        print(f"[Fallback] Train directory not found. Loading dynamically from flat directory: {flat_dir}")
        
        # Read subdirectories as classes
        classes = sorted([c for c in os.listdir(flat_dir) if os.path.isdir(os.path.join(flat_dir, c))])
        if len(classes) == 0:
            raise FileNotFoundError(f"No class folders found in directory: {flat_dir}")
            
        images = []
        labels = []
        for class_idx, class_name in enumerate(classes):
            class_path = os.path.join(flat_dir, class_name)
            img_paths = glob.glob(os.path.join(class_path, "*"))
            print(f"Loading {len(img_paths)} images for class '{class_name}'...")
            for img_path in img_paths:
                try:
                    img = load_and_preprocess_image(img_path, target_size=target_size, use_clahe=True)
                    images.append(img)
                    labels.append(class_idx)
                except Exception as e:
                    pass
                    
        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        num_classes = len(classes)
        labels_onehot = to_categorical(labels, num_classes=num_classes)
        
        # Train / Temp Split (70% Train, 30% Temp)
        x_train, x_temp, y_train, y_temp = train_test_split(images, labels_onehot, test_size=0.3, random_state=42, stratify=labels)
        # Val / Test Split (15% Val, 15% Test)
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, stratify=np.argmax(y_temp, axis=-1))
        
        print(f"Dynamic dataset split: Train={len(x_train)}, Val={len(x_val)}, Test={len(x_test)}")
        
        # Create generators using flow()
        # Since we applied CLAHE during loading, we don't need double CLAHE in custom preprocessing
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            horizontal_flip=True,
            zoom_range=0.15,
            width_shift_range=0.15,
            height_shift_range=0.15,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        val_test_datagen = ImageDataGenerator()
        
        train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True)
        val_generator = val_test_datagen.flow(x_val, y_val, batch_size=batch_size, shuffle=False)
        test_generator = val_test_datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=False)
        
        # Attach helper attributes to match directory generators
        train_generator.num_classes = num_classes
        train_generator.class_indices = {c: i for i, c in enumerate(classes)}
        train_generator.samples = len(x_train)
        
        val_generator.num_classes = num_classes
        val_generator.class_indices = {c: i for i, c in enumerate(classes)}
        val_generator.samples = len(x_val)
        
        test_generator.num_classes = num_classes
        test_generator.class_indices = {c: i for i, c in enumerate(classes)}
        test_generator.samples = len(x_test)
        
        return train_generator, val_generator, test_generator

    # Default flow if split folders exist
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
