import torch
import torchvision
from torchvision.datasets import MNIST

import torchvision.transforms as tt

from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F

import numpy as np


## IMPORTING DATA
dataset = MNIST('image_cla_lr/data', train=True, transform=tt.Compose([tt.ToTensor(), tt.Normalize(mean=(0.5), std=(0.5))]))
val_dataset = MNIST('image_cla_lr/data', train=False, transform=tt.Compose([tt.ToTensor(), tt.Normalize(mean=(0.5), std=(0.5))]))



## CREATING DATALOADERS
batch_size = 256

train_dl = DataLoader(dataset, batch_size, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size, shuffle=True)



## CREATING THE MODEL
def conv_2d(ni, nf, ks=3, stride=1):
  return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=False)

def bn_relu_conv(ni, nf):
  return nn.Sequential(
    nn.BatchNorm2d(ni),
    nn.ReLU(inplace=True),
    conv_2d(ni, nf)
  )

class ResidualBlock(nn.Module):
  def __init__(self, ni, nf, stride=1):
    super(ResidualBlock, self).__init__()

    self.bn = nn.BatchNorm2d(ni)
    self.conv1 = conv_2d(ni, nf, stride=stride)
    self.conv2 = bn_relu_conv(nf, nf)
    self.shortcut = lambda x: x
    if ni != nf:
      self.shortcut = conv_2d(ni, nf, ks=1, stride=stride)
    
  def forward(self, x):
    x = F.relu(self.bn(x), inplace=True)
    r = self.shortcut(x)
    x = self.conv1(x)
    x = self.conv2(x) * 0.2
    return x.add_(r)

def make_group(N, ni, nf, stride):
  start = ResidualBlock(ni, nf, stride)
  rest = [ResidualBlock(nf, nf) for j in range(1, N)]
  return [start] + rest


class WideResNet(nn.Module):
  def __init__(self, start_channel, N, n_classes, groups=3, k=6):
    super().__init__()

    layers = [conv_2d(1, start_channel)]
    channels = [start_channel]

    for i in range(groups):
      channels.append(start_channel*(2**i)*k)
      stride = 2 if i > 0 else 1
      layers += make_group(N, channels[i], channels[i+1], stride)

    layers += [
      nn.BatchNorm2d(channels[-1]),
      nn.ReLU(inplace=True),
      nn.AdaptiveAvgPool2d(1),
      nn.Flatten(),
      nn.Linear(channels[-1], n_classes)
    ]

    self.features = nn.Sequential(*layers)

  def forward(self, x): return self.features(x)


def wdr22():
  return WideResNet(start_channel=16, N=3, n_classes=10, groups=3, k=6)

model = wdr22()



## TRAINING THE MODEL
def loss_batch(model, loss_fn, xb, yb, opt=None, metric=None):
  out = model(xb)
  loss = loss_fn(out, yb)

  if opt is not None:
    loss.backward()
    opt.step()
    opt.zero_grad()

  metric_result = None
  if metric is not None:
    metric_result = metric(out, yb)
  
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

  if opt_fn is None: opt_fn = optim.Adam
  opt = opt_fn(model.parameters(), lr)

  train_losses, val_losses, val_metrics = [], [], []

  for epoch in range(epochs):
    print("NEW EPOCH:", epoch+1, "/", epochs)
    model.train()
    for xb, yb in train_dl:
      train_loss, _, _ = loss_batch(model, loss_fn, xb, yb, opt=opt)

    model.eval()
    val_loss, _, val_metric = evaluate(model, loss_fn, val_dl, metric=metric)
    print(train_loss, val_loss, val_metric)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_metrics.append(val_metric)
  
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
model.to(device)

loss_fn = F.cross_entropy
opt_fn = optim.Adam



fit(2, model, loss_fn, train_dl, val_dl, 5e-3, opt_fn, accuracy)
fit(2, model, loss_fn, train_dl, val_dl, 5e-4, opt_fn, accuracy)

torch.save(model.state_dict(), 'image_cla_lr/models/model8')