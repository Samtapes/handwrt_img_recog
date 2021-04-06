import torch
import torchvision
from torchvision.datasets import MNIST

import torchvision.transforms as Transform

from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

import torch.nn as nn

import torch.nn.functional as F

import numpy as np


# Importing data
dataset = MNIST('image_cla_lr/data', train=True, transform=Transform.ToTensor())
evaluate_dataset = MNIST('image_cla_lr/data', train=False, transform=Transform.ToTensor())





# SPLIT INDICES FOR SAMPLERS FOR DATALOADERS
def split_indices(n, val_pct):
  n_val = int(n * val_pct)
  idxs = np.random.permutation(n)
  return idxs[n_val:], idxs[:n_val]

test_indices, val_indices = split_indices(len(evaluate_dataset), 0.5)





# DATALOADERS
batch_size = 100

train_loader = DataLoader(dataset, batch_size, shuffle=True)


test_sampler = SubsetRandomSampler(test_indices)
test_loader = DataLoader(evaluate_dataset, batch_size, sampler=test_sampler)


val_sampler = SubsetRandomSampler(val_indices)
val_loader = DataLoader(evaluate_dataset, batch_size, sampler=val_sampler)





# MODEL
input_size = 28 * 28
num_classes = 10

class MnistModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(input_size, num_classes)

  def forward(self, xb):
    xb = xb.reshape(-1, input_size)
    out = self.linear(xb)
    return out

model = MnistModel()





# LOSS FUNCTION
loss_fn = F.cross_entropy




# OPTIMIZER
opt = torch.optim.SGD(model.parameters(), lr=1e-3)




# LOSS BATCH FUNCTION
def loss_batch(model, xb, yb, loss_fn, opt=None, metric=None):
  preds = model(xb)
  loss = loss_fn(preds, yb)

  if opt is not None:
    loss.backward()
    opt.step()
    opt.zero_grad()
  
  metric_result = None
  if metric is not None:
    metric_result = metric(preds, yb)

  return loss, len(xb), metric_result




# ACCURACY FUNCTION
def accuracy(out, labels):
  _, pred = torch.max(out, 1)
  return torch.sum(pred == labels).item() / pred.numel()





# EVALUATE FUNCTION
def evaluate(model, loss_fn, val_loader, metric=None):
  results = [loss_batch(model, xb, yb, loss_fn, metric=metric) for xb, yb in val_loader]

  loss, nums, metric_result = zip(*results)

  total = np.sum(nums)

  avg_loss = np.sum(np.multiply(loss, nums)) / total

  avg_metric = None
  if metric is not None:
    avg_metric = np.sum(np.multiply(metric_result, nums)) / total
  
  return avg_loss, total, avg_metric




# TRAINING THE MODEL
def fit(num_epochs, model, train_loader, loss_fn, opt, metric=None, save_model=False):
  for epoch in range(num_epochs):
    print("New epoch:", epoch, "/", num_epochs)
    for xb, yb in train_loader:
      loss, _, metric_result = loss_batch(model, xb, yb, loss_fn, opt, metric)

      if save_model is not False:
        if(loss < 0.39 and metric_result >= 0.93):
          torch.save(model.state_dict(), 'image_cla_lr/models/model4')

    print(loss, metric_result)

fit(100, model, train_loader, loss_fn, opt, accuracy, True)




# EVALUATING THE MODEL
avg_loss, _, avg_accuracy = evaluate(model, loss_fn, val_loader, accuracy)
print(avg_loss, avg_accuracy)




# TESTING THE MODEL
def test(model, test_loader):
  for xb, yb in test_loader:
    out = model(xb)
    _, pred = torch.max(out, 1)
    print(pred[0], yb[0])


test(model, test_loader)