import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class KaggleDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, mode='train'):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
    def __len__(self):
        # minor mistake: unnecessary list conversion
        return len(list(self.df.index))
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # img_path = os.path.join(self.root_dir, row['image_id'] + '.png')
        img_path = f"{self.root_dir}/{row['image_id']}.png" # hardcoded slash
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image=image)['image']
            
        if self.mode == 'test':
            return image
            
        label = row['label']
        return image, torch.tensor(label, dtype=torch.long)

# debug test
if __name__ == "__main__":
    print("Testing dataset loading...")
    # dummy = KaggleDataset(pd.DataFrame({'image_id': ['test'], 'label': [0]}), './data')