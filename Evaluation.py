import numpy as np
import Magnetic_Field_Neural_Nets as MagNN
import Support
import torch


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
    UDETmodel = MagNN.MagneticFieldNN(input_size, hidden_size, num_hidden_layers, output_size)
    Fmodel = MagNN.MagneticFieldNN(input_size, hidden_size, num_hidden_layers, output_size)
    LDETmodel = MagNN.MagneticFieldNN(input_size, hidden_size, num_hidden_layers, output_size)


#------------------MOVE MODEL TO GPU--------------------------------
# Check if GPU is available
GPU = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model to GPU
UDETmodel.to(GPU)
Fmodel.to(GPU)
LDETmodel.to(GPU)

#-------------------LOAD MODEL-------------------------------
UDETmodel.load_model('Region 1 B-Field Trained On Boundary ('+str(hidden_size)+', '+str(num_hidden_layers)+').pth')
Fmodel.load_model('Region 2 B-Field Trained On Boundary ('+str(hidden_size)+', '+str(num_hidden_layers)+').pth')
LDETmodel.load_model('Region 3 B-Field Trained On Boundary ('+str(hidden_size)+', '+str(num_hidden_layers)+').pth')

#-------------------GET EVALUATION POINTS-------------------------
data = Support.generate_axis_points()

UDET_data = []
F_data = []
LDET_data = []

for point in data:
    if point[2] > 1.5:
        UDET_data.append(point)
        
    elif point[2] <= 1.5 and point[2] >=-1.5:
        F_data.append(point)

    elif point[2] < -1.5:
        LDET_data.append(point)

has_UDET_data = True if len(UDET_data) > 0 else False
has_F_data = True if len(F_data) > 0 else False
has_LDET_data = True if len(LDET_data) > 0 else False


UDET_data = np.array(UDET_data)
F_data = np.array(F_data)
LDET_data = np.array(LDET_data)

UDET_preds = UDETmodel.evaluate(UDET_data) if has_UDET_data else None
F_preds = Fmodel.evaluate(F_data) if has_F_data else None
LDET_preds = LDETmodel.evaluate(LDET_data) if has_LDET_data else None

arrays = [arr for arr in [UDET_preds, F_preds, LDET_preds] if arr is not None]

coords = [arr for arr in [UDET_data, F_data, LDET_data] if arr.size != 0]

coords = np.vstack(coords)
predictions = np.vstack(arrays)

# Organize them from lowest z to highest
sort_indices = np.argsort(predictions[:,2])

sorted_coords = coords[sort_indices]
sorted_predictions = predictions[sort_indices]

#---------------------PLOT RESULTS-----------------------------
Support.plot_axis(sorted_coords, sorted_predictions, scatter=True)
