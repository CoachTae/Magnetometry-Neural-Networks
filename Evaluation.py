import os
import sys
import numpy as np
import Magnetic_Field_Neural_Nets as MagNN
import Support
import torch
import time
projects_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ferenc_path = os.path.join(projects_dir, "Nab Magnetometry/Bferenc Files/Magnetometry-master/Bferenc Python/")
sys.path.append(ferenc_path)
magfield_path = os.path.join(ferenc_path, "magfield")
sys.path.append(magfield_path)

from MainRoutine import mainmagfield
import magfield


#--------------PARAMETERS--------------------------------------
# B, A, or phi are available options
field_type = 'B'
compare = True


#---------------MODEL TO BE EVALUATED----------------------------
file = f'{field_type}-Field Trained On Boundary (32, 8).pth'




#----------------NETWORK ARCHITECTURE-----------------------------
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

#---------------------CREATE MODEL-------------------------------
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
predictions = model.evaluate(coords, field_type)

# Organize them from lowest z to highest
sort_indices = np.argsort(coords[:,2])

sorted_coords = coords[sort_indices]
sorted_predictions = predictions[sort_indices]

#---------------------PLOT RESULTS-----------------------------
if compare:
    coords_m = sorted_coords.astype(float) / 100
    actual = Support.load_axis()
    Support.compare_fields(sorted_coords, sorted_predictions, actual)
else:
    Support.plot_axis(sorted_coords, sorted_predictions, scatter=True)
