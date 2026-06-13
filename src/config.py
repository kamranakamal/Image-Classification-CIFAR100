import torch
import torch.nn as nn
from torchmetrics.classification import Accuracy
from .Helper import EarlyStopping
from torchvision import transforms
EPOCHS = 30
BATCH_SIZE = 256
NUM_WORKERS = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
acc_fn = Accuracy(task="multiclass", num_classes=100).to(device)
def get_hparams(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    cosine_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=5, verbose=True)
    return optimizer, cosine_lr, early_stopping
mean, std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=3, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
