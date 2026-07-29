import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from utils.preprocessing import load_and_preprocess_image

# Make sure Custom Layer is mapped during model loading
from models.adaptive_fusion import AdaptiveWeightedFusion
from models.cbam import ChannelAttention, SpatialAttention

def main():
    parser = argparse.ArgumentParser(description="Predict COVID-19 CT Scan Class for a Single Image")
    parser.add_argument('--image_path', type=str, required=True, help='Path to target CT scan image')
    parser.add_argument('--model_path', type=str, default='best_model.h5', help='Path to trained model')
    parser.add_argument('--class_names', type=str, default='COVID-19,Normal,Other', help='Comma-separated class names')
    parser.add_argument('--save_visualization', type=str, default='prediction_output.png', help='Filepath to save visualization')
    args = parser.parse_args()

    class_names = [name.strip() for name in args.class_names.split(',')]
    
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found at path: {args.image_path}")
        
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Trained model not found at path: {args.model_path}. Run train.py first.")

    # 1. Load and Preprocess Image
    print("Loading and preprocessing image...")
    preprocessed_img = load_and_preprocess_image(args.image_path, target_size=(224, 224), use_clahe=True)
    
    # Add batch dimension: (1, 224, 224, 3)
    input_tensor = np.expand_dims(preprocessed_img, axis=0)

    # 2. Load Model
    print(f"Loading trained ADF-Net model from: {args.model_path}...")
    model = tf.keras.models.load_model(
        args.model_path,
        custom_objects={
            'AdaptiveWeightedFusion': AdaptiveWeightedFusion,
            'ChannelAttention': ChannelAttention,
            'SpatialAttention': SpatialAttention
        }
    )

    # 3. Predict probabilities
    print("Running inference...")
    probs = model.predict(input_tensor)[0]
    predicted_idx = np.argmax(probs)
    predicted_label = class_names[predicted_idx]
    confidence = probs[predicted_idx]

    # Print results
    print("\n================ PREDICTION RESULTS ================")
    print(f"Predicted Class:    {predicted_label}")
    print(f"Confidence Score:   {confidence * 100:.2f}%")
    print("Class Probabilities:")
    for idx, prob in enumerate(probs):
        class_name = class_names[idx] if idx < len(class_names) else f"Class {idx}"
        print(f" - {class_name}: {prob * 100:.2f}%")
    print("====================================================\n")

    # 4. Save and Show Visualization
    # Read original image for background display
    orig_img = cv2.imread(args.image_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 5))
    
    # Left subplot: Original Image with prediction text
    plt.subplot(1, 2, 1)
    plt.imshow(orig_img)
    plt.title("Input CT Scan")
    plt.axis("off")
    plt.text(
        10, 30, 
        f"Prediction: {predicted_label}\nConfidence: {confidence*100:.1f}%", 
        color='cyan', 
        fontsize=12,
        weight='bold',
        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.5')
    )
    
    # Right subplot: Confidence probability bar chart
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(class_names))
    colors = ['red' if idx == predicted_idx else 'grey' for idx in range(len(class_names))]
    plt.barh(y_pos, probs, align='center', alpha=0.8, color=colors)
    plt.yticks(y_pos, class_names)
    plt.xlim(0, 1.05)
    plt.xlabel('Probability')
    plt.title('Prediction Confidence Probability')
    
    plt.tight_layout()
    plt.savefig(args.save_visualization, dpi=300)
    plt.close()
    
    print(f"Prediction visualization successfully saved to: {args.save_visualization}")

if __name__ == '__main__':
    main()
