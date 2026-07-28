import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, classification_report
)

def calculate_specificity_multiclass(y_true, y_pred, num_classes):
    """
    Computes class-wise specificity and overall average specificity for multiclass classification.
    Specificity = TN / (TN + FP)
    """
    cm = confusion_matrix(y_true, y_pred)
    specificities = []
    
    for i in range(num_classes):
        # True Positives
        tp = cm[i, i]
        # False Positives (column sum - TP)
        fp = np.sum(cm[:, i]) - tp
        # False Negatives (row sum - TP)
        fn = np.sum(cm[i, :]) - tp
        # True Negatives (total sum - row sum - col sum + TP)
        tn = np.sum(cm) - (tp + fp + fn)
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(specificity)
        
    return np.array(specificities), np.mean(specificities)

def compute_all_metrics(y_true, y_pred, y_prob, num_classes, target_names=None):
    """
    Computes all classification metrics: Accuracy, Precision, Recall, F1, Specificity, AUC.
    Returns a dictionary of results.
    """
    # Convert one-hot encoded outputs to class labels if needed
    if len(y_true.shape) > 1 and y_true.shape[-1] > 1:
        y_true_lbl = np.argmax(y_true, axis=-1)
    else:
        y_true_lbl = y_true

    if len(y_pred.shape) > 1 and y_pred.shape[-1] > 1:
        y_pred_lbl = np.argmax(y_pred, axis=-1)
    else:
        y_pred_lbl = y_pred

    # Accuracy
    acc = accuracy_score(y_true_lbl, y_pred_lbl)
    
    # Precision, Recall, F1 (weighted averages)
    prec = precision_score(y_true_lbl, y_pred_lbl, average='weighted', zero_division=0)
    rec = recall_score(y_true_lbl, y_pred_lbl, average='weighted', zero_division=0)
    f1 = f1_score(y_true_lbl, y_pred_lbl, average='weighted', zero_division=0)
    
    # Specificity
    class_specs, avg_spec = calculate_specificity_multiclass(y_true_lbl, y_pred_lbl, num_classes)
    
    # ROC AUC
    try:
        if len(y_prob.shape) > 1 and y_prob.shape[-1] > 1:
            # Multiclass case
            auc = roc_auc_score(y_true, y_prob, average='weighted', multi_class='ovr')
        else:
            # Binary/1D prob case
            auc = roc_auc_score(y_true_lbl, y_prob)
    except Exception:
        auc = 0.0
        
    # Confusion Matrix
    cm = confusion_matrix(y_true_lbl, y_pred_lbl)
    
    # Text Report
    report = classification_report(y_true_lbl, y_pred_lbl, target_names=target_names, digits=4, zero_division=0)
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "class_specificity": class_specs,
        "average_specificity": avg_spec,
        "auc_score": auc,
        "confusion_matrix": cm,
        "classification_report": report
    }
