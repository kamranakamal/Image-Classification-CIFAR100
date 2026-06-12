
from tqdm.auto import tqdm
import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False

    def check_early_stop(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")
                return self.stop_training


class Trainer:
  def __init__(self, model, train_loader,test_loader, loss_fn,metric, device):
    self.model = model
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.loss_fn = loss_fn
    self.metric = metric
    self.metric.name = type(self.metric).__name__
    self.history = {}
    self.history["Train"] = []
    self.history["Test"] = []
    self.min_loss = np.inf
    self.device = device

  def train_step(self, optimizer):
    self.model.train()
    total_loss = 0
    for X, y in tqdm(self.train_loader, desc="Training", leave=False):
      X, y = X.to(self.device), y.to(self.device)
      logits = self.model(X)
      preds = torch.argmax(logits, dim=1)

      loss = self.loss_fn(logits, y)
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(
        self.model.parameters(),
        max_norm=1.0)
      optimizer.step()

      total_loss+=loss.item()
      total_metric = self.metric(preds, y)

    total_loss /= len(self.train_loader)
    total_metric = self.metric.compute().item()
    self.metric.reset()
    return total_loss, total_metric



  def eval_step(self, loader=None, *metrics):
    loader = loader if loader is not None else self.test_loader
    self.model.eval()
    total_loss = 0
    if len(metrics) == 0:
      metrics = (self.metric,)

    for metric in metrics:
        metric.reset()
    with torch.inference_mode():
      for X, y in tqdm(loader, desc='Evaluating', leave=False):
        X, y = X.to(self.device), y.to(self.device)
        logits = self.model(X)
        preds = torch.argmax(logits, dim=1)

        loss = self.loss_fn(logits, y)
        total_loss+=loss.item()

        for metric in metrics:
          metric.update(preds, y)
    total_loss /= len(loader)
    metric_dict = {}

    for metric in metrics:
      metric_name = type(metric).__name__
      metric_dict[metric_name] = metric.compute().item()
      metric.reset()
    return total_loss, metric_dict



  def train(self, epochs:int, optimizer, scheduler_fn=None, train_only=True,stopping_fn=None):
    self.model.to(self.device)
    for epoch in tqdm(range(1, epochs+1), desc="Epoch"):
      train_loss, train_metric = self.train_step(optimizer)
      val_loss = train_loss
      self.history["Train"].append({"Epoch":epoch,"Loss":train_loss, self.metric.name:train_metric})
      msg = (
        f"Epoch: {epoch} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train {self.metric.name}: {train_metric:.4f}"
        )

      if not train_only:
        test_loss, test_metric_dict = self.eval_step(loader=None)
        val_loss = test_loss
        self.history['Test'].append({"Epoch": epoch, "Loss":test_loss, self.metric.name:test_metric_dict[self.metric.name]})
        msg += (
            f" | Test Loss: {test_loss:.4f}"
            f" | Test {self.metric.name}: "
            f"{test_metric_dict[self.metric.name]:.4f}"
            )
      print(msg)

      if scheduler_fn is not None:
        scheduler_fn.step(train_loss)
      if stopping_fn is not None:
        stop_training = stopping_fn.check_early_stop(val_loss)
        if stop_training:
          return self.history
      if val_loss < self.min_loss:
        self.min_loss = train_loss
        torch.save(self.model.state_dict(),f"{epoch}_epoch_{val_loss:.4f}_val_loss.pth")
    return self.history

  @property
  def plot_history(self):
    train_history_df = pd.DataFrame(self.history['Train'])
    test_history_df = pd.DataFrame(self.history['Test'])
    if len(train_history_df)==0:
      raise ValueError("No History is Saved, Train the model and save the history first.....")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].plot(train_history_df['Epoch'], train_history_df['Loss'], label="Train Loss")
    axes[1].plot(train_history_df['Epoch'], train_history_df[self.metric.name], label=f"Train {self.metric.name}")
    if len(train_history_df) == len(test_history_df):
      axes[0].plot(test_history_df['Epoch'], test_history_df["Loss"], label='Test Loss')
      axes[1].plot(test_history_df['Epoch'], test_history_df[self.metric.name], label=f"Test {self.metric.name}")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss at each epoch")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel(self.metric.name)
    axes[1].set_title(f"{self.metric.name} at each epoch")
    axes[1].legend()
    axes[0].legend()


def create_loaders(train_ds,
                   test_ds,
                   train_transform=None,
                   test_transform=None,
                   batch_size=64,
                   num_workers=0,
                   pin_memory=True,
                   persistent_workers=True):
    if train_transform and test_transform:
        train_ds.transform = train_transform
        test_ds.transform = test_transform
    train_loader = DataLoader(train_ds,batch_size, True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    test_loader = DataLoader(test_ds, batch_size, False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    return train_loader, test_loader


def load_cifar_classes(filename):
    class_names = []
    
    with open(filename, 'r') as f:
        for line in f:
            # Strip whitespace (newlines) and ignore empty lines
            name = line.strip()
            if name:
                class_names.append(name)
                
    return class_names

