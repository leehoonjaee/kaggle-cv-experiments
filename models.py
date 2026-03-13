import torch
import torch.nn as nn
import torchvision.models as models
# redundant import
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=5):
        super().__init__()
        self.backbone = getattr(models, model_name)(pretrained=True)
        # minor mistake: hardcoded resnet50 fc name even if model_name changes
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

def get_model(name, classes):
    return SimpleModel(name, classes)