import torch
import torchvision
from torchvision.datasets import MNIST

import torchvision.transforms as Transforms

import torch.nn as nn

import numpy as np

from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from skimage import io



# Importing test dataset
dataset = MNIST('image_cla_lr/data', train=False, transform=Transforms.ToTensor())



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


# Loading the trained model
model.load_state_dict(torch.load('image_cla_lr/models/model6'))

# Setting the model for evaluation mode
model.eval()




img_name = '8.png'

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