import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
import random as rand
import Magnetic_Field_Neural_Nets as MagNN
import Support
#import wandb

#---------------INITIALIZE WEIGHTS AND BIASES LOGGING----------------------
#wandb.init(project='Magnetometry NN')


#--------------PARAMETERS---------------------------------------------------
#B, A, or phi are available options
field_type = 'B'

# Weights for the MSE Loss, Divergence Loss, Curl Loss, L1 Reg, and Laplacian respectively
lambdas = [2, 1, 1, 1e-5, 100]

# What data to train the model on
    # Boundary trains on just the box-cylinder's boundary
data_type = 'Boundary'

# Regularization
L1 = True

#--------------NETWORK ARCHITECTURE-----------------------------------------
if field_type == 'B' or field_type == 'A':
    input_size = 3 # x, y, z coordinates
    hidden_size = 32 # Neurons per layer
    num_hidden_layers = 8
    output_size = 3 # Bx, By, Bz

elif field_type == 'phi':
    input_size = 3 # x, y, z coordinates
    hidden_size = 32 # Neurons per layer
    num_hidden_layers = 8
    output_size = 1 # Scalar value


#--------------CREATE MODEL OBJECT---------------------------------------------------
model = MagNN.MagneticFieldNN(input_size, hidden_size, num_hidden_layers, output_size)


#-------------MOVE MODEL TO GPU--------------------------------------------
# Check if GPU is available
GPU = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model to GPU
model.to(GPU)


#--------------LOAD DATA---------------------------------------------------
'''if data_type == 'Boundary':
    top_cap, bot_cap, shell = Support.load_boundary(region)

if data_type == 'Axis':
    data = Support.load_boundary()'''


#-------------TRAIN THE MODEL-----------------------------------------------
#model.interior_scan(num_points_per_scan=1000,
                        #num_scans=100, region=region)

#print("Done with initial interior scan.")
#print("\n\nTraining the model.\n\n")

model.train_model(num_epochs=50,
                  num_points=10000,
                  train_split=0.8,
                  validation_threshold=1e-9,
                  int_point_ratio=10,
                  dataType=data_type,
                  batch_size=64,
                  lambdas=lambdas,
                  L1=L1,
                  Field=field_type
                  )

model.save_model(f'{field_type}-Field Trained On Boundary ({hidden_size}, {num_hidden_layers}).pth')
