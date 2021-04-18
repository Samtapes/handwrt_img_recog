import torch
import torchvision
from torchvision.datasets import MNIST

from torchvision.transforms import ToTensor

import numpy as np

from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader

import torch.nn as nn

import torch.nn.functional as F


## IMPORTING DATASET
dataset = MNIST('image_cla_lr/data', train=True, transform=ToTensor())
evaluate_dataset = MNIST('image_cla_lr/data', train=False, transform=ToTensor())




## SPLITING INDICES
def split_indices(n, val_pct, seed=99):
  n_val = int(n * val_pct)

  np.random.seed(seed)

  idxs = np.random.permutation(n)

  return idxs[n_val:], idxs[:n_val]

test_indices, val_indices = split_indices(len(evaluate_dataset), 0.9, 42)




## CREATING DATALOADERS
batch_size = 100

train_dl = DataLoader(dataset, batch_size, shuffle=True)


val_sampler = SubsetRandomSampler(val_indices)
val_dl = DataLoader(evaluate_dataset, batch_size, sampler=val_sampler)


test_sampler = SubsetRandomSampler(test_indices)
test_dl = DataLoader(evaluate_dataset, batch_size, sampler=test_sampler)




## MODEL
model = nn.Sequential(
  nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
  nn.ReLU(),
  nn.AvgPool2d(2,2), # bc, 16, 14, 14 

  nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
  nn.ReLU(),
  nn.AvgPool2d(2,2), # bc, 32, 7, 7

  nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
  nn.ReLU(),
  nn.AvgPool2d(2,2), # bc, 64, 3.5, 3.5

  nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
  nn.ReLU(),
  nn.AvgPool2d(2,2),

  nn.Flatten(),
  nn.Linear(128, 10)
)




## TRAINING THE MODEL
def loss_batch(model, loss_fn, xb, yb, opt=None, metric=None):
  pred = model(xb)
  loss = loss_fn(pred, yb)

  if opt is not None:
    loss.backward()
    opt.step()
    opt.zero_grad()

  metric_result = None
  if metric is not None:
    metric_result = metric(pred, yb)

  return loss.item(), len(xb), metric_result


def accuracy(out, yb):
  _, pred = torch.max(out, 1)
  return torch.sum(pred == yb).item() / len(pred)


def evaluate(model, loss_fn, val_dl, metric=None):
  with torch.no_grad():
    results = [loss_batch(model, loss_fn, xb, yb, metric=metric) for xb, yb in val_dl]

    losses, nums, metrics = zip(*results)

    total = np.sum(nums)

    avg_loss = np.sum(np.multiply(losses, nums)) / total

    avg_metric = None
    if metric is not None:
      avg_metric = np.sum(np.multiply(metrics, nums)) / total
    
  return avg_loss, total, avg_metric



def fit(epochs, model, loss_fn, train_dl, val_dl, lr, opt_fn=None, metric=None):

  if opt_fn is None: opt_fn = torch.optim.SGD
  opt = opt_fn(model.parameters(), lr=lr)

  train_losses, val_losses, val_metrics = [], [], []

  for epoch in range(epochs):
    print("NEW EPOCH:", epoch + 1, "/", epochs)
    model.train()
    for xb, yb in train_dl:
      train_loss, _, _ = loss_batch(model, loss_fn, xb, yb, opt)
    
    # EVALUTING
    model.eval()
    avg_loss, _, avg_metric = evaluate(model, loss_fn, val_dl, metric)
    print(avg_loss, avg_metric)

    train_losses.append(train_loss)
    val_losses.append(avg_loss)
    val_metrics.append(avg_metric)
  
  return train_losses, val_losses, val_metrics




# Passing everything to GPU if there is one
def get_default_device():
  if torch.cuda.is_available():
    return torch.device('cuda')
  else:
    return torch.device('cpu')


device = get_default_device()



def to_device(data, device):
  if isinstance(data, (list, tuple)):
    return [to_device(x, device) for x in data]
  return data.to(device, non_blocking=True)



class DeviceDataLoader():
  def __init__(self, dl, device):
    self.dl = dl
    self.device = device

  def __iter__(self):
    for b in self.dl:
      yield to_device(b, self.device)
    
  def __len__(self):
    return len(self.dl)

train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
test_dl = DeviceDataLoader(test_dl, device)

model.to(device)



loss_fn = F.cross_entropy

opt_fn = torch.optim.SGD

fit(20, model, loss_fn, train_dl, val_dl, 0.05, opt_fn, accuracy)

torch.save(model.state_dict(), 'image_cla_lr/models/model6')