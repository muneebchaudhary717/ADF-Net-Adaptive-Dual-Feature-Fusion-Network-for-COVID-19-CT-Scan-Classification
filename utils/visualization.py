import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Set publication quality styles
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14
})

def plot_training_history(history, save_dir):
    """
    Plots training and validation loss and accuracy curves.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Accuracy Curve
    plt.figure(figsize=(8, 5))
    acc = history.history.get('acc', history.history.get('accuracy'))
    val_acc = history.history.get('val_acc', history.history.get('val_accuracy'))
    epochs = range(1, len(acc) + 1)
    
    plt.plot(epochs, acc, '#2b5c8f', label='Training Accuracy', linewidth=2)
    if val_acc:
        plt.plot(epochs, val_acc, '#d97d24', label='Validation Accuracy', linewidth=2)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_accuracy.png'), dpi=300)
    plt.close()

    # 2. Loss Curve
    plt.figure(figsize=(8, 5))
    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')
    
    plt.plot(epochs, loss, '#2b5c8f', label='Training Loss', linewidth=2)
    if val_loss:
        plt.plot(epochs, val_loss, '#d97d24', label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_loss.png'), dpi=300)
    plt.close()

def plot_confusion_matrix(cm, class_names, save_dir):
    """
    Plots a publication-quality confusion matrix heatmap.
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(7, 6))
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names,
        cbar=True,
        square=True,
        annot_kws={"size": 12}
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

def plot_roc_curve(y_true, y_prob, num_classes, class_names, save_dir):
    """
    Plots One-vs-Rest Multiclass ROC curves.
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    
    # Calculate ROC curves per class
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.4f})', linewidth=2)
        
    plt.plot([0, 1], [0, 1], 'k--', label='Chance Level', linewidth=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300)
    plt.close()

def plot_precision_recall_curve(y_true, y_prob, num_classes, class_names, save_dir):
    """
    Plots One-vs-Rest Precision-Recall curves.
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    
    # Calculate PR curves per class
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
        avg_precision = average_precision_score(y_true[:, i], y_prob[:, i])
        plt.plot(recall, precision, label=f'{class_names[i]} (AP = {avg_precision:.4f})', linewidth=2)
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multiclass Precision-Recall (PR) Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'precision_recall_curve.png'), dpi=300)
    plt.close()
