import numpy as np
import Magnetic_Field_Neural_Nets as MagNN
import Support
import torch

#--------------PARAMETERS--------------------------------------
# B, A, or phi are available options
field_type = 'B'

# What region the model will look at
    # 1 = UDET
    # 2 = Filter
    # 3 = LDET
region = 1

#----------------NETWORK ARCHITECTURE-----------------------------
if field_type == 'B' or field_type == 'A':
    input_size = 3 # x, y, z coordinates
    hidden_size = 32 # Neurons per layer
    num_hidden_layers = 4
    output_size = 3 # Bx, By, Bz

elif field_type == 'phi':
    input_size = 3 # x, y, z coordinates
    hidden_size = 32 # Neurons per layer
    num_hidden_layers = 4
    output_size = 1 # Scalar value

#---------------------CREATE MODEL-------------------------------
if field_type == 'B':
    model = MagNN.MagneticFieldNN(input_size, hidden_size, num_hidden_layers, output_size)


#------------------MOVE MODEL TO GPU--------------------------------
# Check if GPU is available
GPU = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model to GPU
model.to(GPU)

#-------------------LOAD MODEL-------------------------------
model.load_model('Region 1 B-Field Trained On Boundary ('+str(hidden_size)+', '+str(num_hidden_layers)+').pth')

#-------------------GET EVALUATION POINTS-------------------------
data = Support.generate_axis_points()

predictions = model.evaluate(data)

#---------------------PLOT RESULTS-----------------------------
Support.plot_axis(data, predictions)
