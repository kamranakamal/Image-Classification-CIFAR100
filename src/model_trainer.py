import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100
from torchvision import transforms
from .Helper import Trainer, EarlyStopping, create_loaders
from .effnet_model import create_effnet
from torchmetrics.classification import Accuracy, MulticlassF1Score
from .config import loss_fn, acc_fn, device, EPOCHS, NUM_WORKERS, BATCH_SIZE,  train_transforms, test_transforms, get_hparams

train_ds = CIFAR100(root='data/', train=True, download=True, transform=train_transforms)
test_ds = CIFAR100(root='data/', train=False, download=True, transform=test_transforms)
class_names = train_ds.classes
with open("classes.txt", "w") as f:
    f.write(str(class_names))

train_loader, test_loader = create_loaders(train_ds, test_ds, train_transform=None, test_transform=None, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

model = create_effnet(device)
optimizer, cosine_lr, early_stopping = get_hparams(model)

trainer = Trainer(model, train_loader, test_loader, loss_fn, acc_fn,device)
trainer.train(EPOCHS, optimizer, cosine_lr, train_only=True, stopping_fn=early_stopping)


