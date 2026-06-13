import torchvision
import torch
from torchvision import transforms
import torch.nn as nn
from .config import device
def create_effnet(device=device, weights=True):
    if weights:
        effnet_weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        effnet_model = torchvision.models.efficientnet_b0(weights=effnet_weights).to(device)
        effnet_transforms = effnet_weights.transforms()
        for params in effnet_model.features.parameters():
            params.requires_grad = False
        for params in effnet_model.features[-3:].parameters():
            params.requires_grad = True
    else:
        effnet_model = torchvision.models.efficientnet_b0().to(device)
    effnet_model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features=1280, out_features=100, bias=True)
    ).to(device);
    return effnet_model
