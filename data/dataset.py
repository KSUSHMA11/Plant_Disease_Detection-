import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from .transforms import get_train_transforms, get_val_transforms
import torch

class PlantDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['healthy', 'multiple_diseases', 'rust', 'scab']
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        # Assuming image_id in CSV matches filename without extension, based on previous observation (Train_0)
        # But list_dir showed Train_0.jpg.
        img_name = row['image_id'] + '.jpg'
        img_path = os.path.join(self.root_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        
        # Get label (argmax of the one-hot columns)
        label_vec = row[self.classes].values.astype('float32')
        label = label_vec.argmax()
        
        if self.transform:
            image = self.transform(image)
            
        return image, int(label)

class TransformSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.subset)

def get_datasets(data_dir, split_ratios=(0.8, 0.1, 0.1)):
    # data_dir should be the 'dataset' folder containing train.csv and images/
    csv_path = os.path.join(data_dir, 'train.csv')
    img_dir = os.path.join(data_dir, 'images')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Images directory not found: {img_dir}")
    
    df = pd.read_csv(csv_path)
    
    # Pass transform=None here, apply later via wrapper to handle train/val split transforms
    full_dataset = PlantDataset(df, img_dir, transform=None)
    
    train_size = int(split_ratios[0] * len(full_dataset))
    val_size = int(split_ratios[1] * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    # Handle rounding issues
    if train_size + val_size + test_size != len(full_dataset):
        test_size = len(full_dataset) - train_size - val_size

    train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])
    
    train_ds = TransformSubset(train_ds, get_train_transforms())
    val_ds = TransformSubset(val_ds, get_val_transforms())
    test_ds = TransformSubset(test_ds, get_val_transforms())
    
    return train_ds, val_ds, test_ds

def get_dataloaders(data_dir, batch_size=32, num_workers=2):
    train_ds, val_ds, test_ds = get_datasets(data_dir)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
