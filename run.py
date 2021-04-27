import torch
import torchvision
from torchvision.datasets import MNIST

import torchvision.transforms as Transforms

import torch.nn as nn

import numpy as np

from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import torch.nn.functional as F

from skimage import io



# Importing test dataset
dataset = MNIST('image_cla_lr/data', train=False, transform=Transforms.ToTensor())



## MODEL
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


# Loading the trained model
model.load_state_dict(torch.load('image_cla_lr/models/model8'))

# Setting the model for evaluation mode
model.eval()




img_name = '7.png'

# Importing image
img1 = io.imread('image_cla_lr/data/images/' + str(img_name), as_gray=True)

# Converting the image to a matrix
input_img_data = np.array(img1)

# Converting to tensor
input_img_data = torch.from_numpy(input_img_data)

# Adding one dimension 
input_img_data = input_img_data.unsqueeze(0)

# Conveting the image to float
input_img_data = input_img_data.type(torch.float32)


# Creating the image label
input_img_label = torch.tensor([int(img_name[0])])


# Normalizing the image
def normalize(img):
  return img - 0.5 / 0.5

input_img_data = normalize(input_img_data)




# Creating the test_loader

# Batch size
batch_size = 1

# Creating dataloader
test_dataset = TensorDataset(input_img_data, input_img_label)
test_loader = DataLoader(test_dataset, batch_size, True)
# test_loader = DataLoader(dataset, batch_size, True)





# Function to make predictions
def test(model, test_loader):
  for xb, yb in test_loader:
    xb = xb.unsqueeze(0)
    out = model(xb)
    _, pred = torch.max(out, 1)
    print(pred[0], yb[0])


test(model, test_loader)