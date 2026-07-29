import os
import argparse
import numpy as np
import tensorflow as tf
from utils.preprocessing import get_data_generators
from utils.metrics import compute_all_metrics
from utils.visualization import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve

# Make sure Custom Layer is registered during model loading
from models.adaptive_fusion import AdaptiveWeightedFusion
from models.cbam import ChannelAttention, SpatialAttention

def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained ADF-Net Model on Test Data")
    parser.add_argument('--dataset_dir', type=str, default='dataset', help='Path to dataset directory')
    parser.add_argument('--model_path', type=str, default='best_model.h5', help='Filepath of the trained model')
    parser.add_argument('--plots_dir', type=str, default='results', help='Directory to save evaluation plots')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    args = parser.parse_args()

    train_dir = os.path.join(args.dataset_dir, 'train')
    val_dir = os.path.join(args.dataset_dir, 'validation')
    if not os.path.exists(val_dir) and os.path.exists(os.path.join(args.dataset_dir, 'val')):
        val_dir = os.path.join(args.dataset_dir, 'val')
    test_dir = os.path.join(args.dataset_dir, 'test')

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Trained model not found at path: {args.model_path}. Run train.py first.")

    # Load Data Generators (Test generator will be used for evaluation)
    print("Loading test generator...")
    _, _, test_gen = get_data_generators(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=args.batch_size,
        target_size=(224, 224),
        dataset_dir=args.dataset_dir
    )

    num_classes = test_gen.num_classes
    class_names = list(test_gen.class_indices.keys())

    # Load Model with custom objects mapping
    print(f"Loading trained ADF-Net model from: {args.model_path}...")
    model = tf.keras.models.load_model(
        args.model_path,
        custom_objects={
            'AdaptiveWeightedFusion': AdaptiveWeightedFusion,
            'ChannelAttention': ChannelAttention,
            'SpatialAttention': SpatialAttention
        }
    )

    print("Running inference on test dataset...")
    # Get true labels and run predictions
    y_true_onehot = []
    y_pred_probs = []

    # Reset generator to prevent index issues
    test_gen.reset()
    
    # Iterate over test generator to get ground truth and predictions
    num_steps = int(np.ceil(test_gen.samples / args.batch_size))
    for _ in range(num_steps):
        x_batch, y_batch = next(test_gen)
        probs_batch = model.predict(x_batch, verbose=0)
        
        y_true_onehot.extend(y_batch)
        y_pred_probs.extend(probs_batch)

    y_true_onehot = np.array(y_true_onehot)
    y_pred_probs = np.array(y_pred_probs)
    y_pred_labels = np.argmax(y_pred_probs, axis=-1)
    y_true_labels = np.argmax(y_true_onehot, axis=-1)

    # Compute Metrics
    print("Computing metrics...")
    metrics = compute_all_metrics(
        y_true=y_true_onehot,
        y_pred=y_pred_labels,
        y_prob=y_pred_probs,
        num_classes=num_classes,
        target_names=class_names
    )

    # Display Metrics Summary
    print("\n================ EVALUATION METRICS ================")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1-Score:    {metrics['f1_score']:.4f}")
    print(f"Specificity: {metrics['average_specificity']:.4f}")
    print(f"ROC AUC:     {metrics['auc_score']:.4f}")
    print("\nClassification Report:")
    print(metrics['classification_report'])
    print("====================================================")

    # Generate and save publication quality plots
    print(f"Saving publication-quality figures to: {args.plots_dir}...")
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, args.plots_dir)
    plot_roc_curve(y_true_onehot, y_pred_probs, num_classes, class_names, args.plots_dir)
    plot_precision_recall_curve(y_true_onehot, y_pred_probs, num_classes, class_names, args.plots_dir)
    
    print("Evaluation pipeline successfully completed.")

if __name__ == '__main__':
    main()
