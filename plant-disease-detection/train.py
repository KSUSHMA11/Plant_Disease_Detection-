import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time

from data.dataset import get_dataloaders
from models import get_vit_model, get_swin_model
from utils.metrics import calculate_metrics

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        # Collect for metrics (optional for train, but good to have)
        all_preds.append(outputs.detach())
        all_targets.append(labels.detach())
        
        pbar.set_postfix({'loss': loss.item()})
        
    epoch_loss = running_loss / len(loader.dataset)
    
    # Calculate train metrics (approximate as we iterate)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = calculate_metrics(all_targets, all_preds)
    metrics['loss'] = epoch_loss
    
    return metrics

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            all_preds.append(outputs)
            all_targets.append(labels)
            
    epoch_loss = running_loss / len(loader.dataset)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = calculate_metrics(all_targets, all_preds)
    metrics['loss'] = epoch_loss
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Plant Disease Detection Training")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--model', type=str, choices=['vit', 'swin'], default='vit', help='Model architecture')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=38, help='Number of classes')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--dry_run', action='store_true', help='Run a single pass for testing')

    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    # Create checkpoint dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(args.data_dir, batch_size=args.batch_size)
    
    # Model
    print(f"Initializing {args.model} model...")
    if args.model == 'vit':
        model = get_vit_model(num_classes=args.num_classes)
    else:
        model = get_swin_model(num_classes=args.num_classes)
        
    if model is None:
        print("Failed to initialize model. Exiting.")
        return

    model = model.to(args.device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_acc = 0.0
    
    epochs = 1 if args.dry_run else args.epochs
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, args.device)
        print(f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f}")
        
        val_metrics = validate(model, val_loader, criterion, args.device)
        print(f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} F1: {val_metrics['f1']:.4f}")
        
        scheduler.step()
        
        # Save best model
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            save_path = os.path.join(args.checkpoint_dir, f"best_model_{args.model}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
            
    print("Training complete.")

if __name__ == "__main__":
    main()
