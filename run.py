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



# MODEL

# All the pixels of the image that are going to enter in the model
input_size = 28 * 28

# All the propalities from 0 to 9
num_classes = 10


# The model
class MnistModel(nn.Module):

  # initing the module
  def __init__(self):
    super().__init__()

    # Model
    self.linear = nn.Linear(input_size, num_classes)
  
  # Insert data in the model
  def forward(self, xb):

    # Reshaping image matrix
    xb = xb.reshape(-1, input_size)

    # Inserting it in the model
    out = self.linear(xb)
    return out


# Seting the model
model = MnistModel()

# Loading the trained model
model.load_state_dict(torch.load('image_cla_lr/models/model4'))

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
    out = model(xb)
    _, pred = torch.max(out, 1)
    print(pred[0], yb[0])


test(model, test_loader)