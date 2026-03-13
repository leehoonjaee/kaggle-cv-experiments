import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# global configs
CFG = {
    'lr': 1e-4,
    'epochs': 10,
    'batch_size': 16,
    'model': 'resnet50'
}

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    pbar = tqdm(loader, total=len(loader))
    total_loss = 0
    
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_description(f"Loss: {loss.item():.4f}")
        
    return total_loss / len(loader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # minor mistake: lr defined in CFG but hardcoded below by accident once
    # model = get_model(CFG['model'], 5).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=0.001) 
    
    # Fixed it but left old code commented out
    # ...
    pass

if __name__ == "__main__":
    # plt.show() # minor mistake: forgot to remove GUI call in headless script
    main()