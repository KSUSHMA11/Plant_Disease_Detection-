import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_metrics(y_true, y_pred):
    """
    Calculates Accuracy, Precision, Recall, and F1-score.
    y_true: tensor of ground truth labels
    y_pred: tensor of model logits
    """
    # Get predictions from logits
    preds = torch.argmax(y_pred, dim=1).cpu().numpy()
    targets = y_true.cpu().numpy()
    
    acc = accuracy_score(targets, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='macro', zero_division=0)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
