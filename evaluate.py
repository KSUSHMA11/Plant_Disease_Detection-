import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from data.dataset import get_dataloaders
from models import get_vit_model, get_swin_model

def plot_confusion_matrix(cm, classes, save_path):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Data (only test split needed)
    _, _, test_loader = get_dataloaders(args.data_dir, batch_size=args.batch_size)
    
    # Initialize Model
    print(f"Loading {args.model} model...")
    if args.model == 'vit':
        model = get_vit_model(num_classes=args.num_classes, pretrained=False)
    else:
        model = get_swin_model(num_classes=args.num_classes, pretrained=False)
        
    model = model.to(device)
    
    # Load Checkpoint
    if os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {args.checkpoint}")
    else:
        print(f"Checkpoint not found at {args.checkpoint}. Testing with random weights.")
        
    model.eval()
    
    all_preds = []
    all_targets = []
    
    print("Evaluating...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    # Metrics
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds))
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plot_confusion_matrix(cm, range(args.num_classes), 'confusion_matrix.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model', type=str, choices=['vit', 'swin'], default='vit')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=38)
    
    args = parser.parse_args()
    evaluate(args)
