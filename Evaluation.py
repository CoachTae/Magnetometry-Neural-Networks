import numpy as np
import Magnetic_Field_Neural_Nets as MagNN
import Support
import torch


#---------------MODEL TO BE EVALUATED----------------------------
file = 'Unified Regions B-Field Trained On Boundary (32, 8).pth'


#--------------PARAMETERS--------------------------------------
# B, A, or phi are available options
field_type = 'B'


#----------------NETWORK ARCHITECTURE-----------------------------
if field_type == 'B' or field_type == 'A':
    input_size = 3 # x, y, z coordinates
    hidden_size = 32 # Neurons per layer
    num_hidden_layers = 8
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
model.load_model(file)

#-------------------GET EVALUATION POINTS-------------------------
coords = Support.generate_axis_points()


#---------------------MAKE PREDICTIONS---------------------------------
predictions = model.evaluate(coords)


# Organize them from lowest z to highest
sort_indices = np.argsort(predictions[:,2])

sorted_coords = coords[sort_indices]
sorted_predictions = predictions[sort_indices]

#---------------------PLOT RESULTS-----------------------------
Support.plot_axis(sorted_coords, sorted_predictions, scatter=True)
