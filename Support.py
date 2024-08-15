import torch
import numpy as np
import matplotlib.pyplot as plt
import math

# Region 1 will be anything above the filter region
# Region 2 will be the filter region, very small
# Region 3 will be anything below the filter region
# This function is for use in the neural network's interior scan to ensure that any given point within a region obeys Maxwell's Equations
def random_points(num_points, region=1, device='cuda'):
    if region == 1: # UDET Region
        radius = 10
        bottom_height = 3
        height = 525 - bottom_height
    
    
    elif region == 2: # F coil region
        radius = 4.5
        bottom_height = -3
        height = 3 - bottom_height
    
    
    elif region == 3: # LDET region
        radius = 10
        bottom_height = -125
        height = -3 - bottom_height

    # Random radial distances
    # The sqrt pushes points outwards to avoid clustering near the center and encourage even distribution in a circle
    r = torch.sqrt(torch.rand(num_points, device=device))*radius

    # Random angles
    theta = torch.rand(num_points, device=device)*2*np.pi

    # Random heights
    z = torch.rand(num_points, device=device)*height + bottom_height

    # Convert to cartesian coordinates
    x = r*torch.cos(theta)
    y = r*torch.sin(theta)

    # Stack coordinates
    points = torch.stack([x, y, z], dim=1)

    return points
        


#-----------------------------BOUNDARY DATA FUNCTIONS------------------------------------------------------------
def generate_boundary_coordinates(dr, bottom, top, num_points, filenames):
    '''
    Generates the coordinates on the surface of a rectangular box.

    dr gives the total width along x and y.
        e.g. dr = 20 means x spans from -10 to +10, and y does the same

    bottom is the z-coordinate corresponding to the bottom of the box

    top is the z-coordinate corresponding to the top of the box

    num_points is the number of points for the entire surface

    filenames should be a list like [bot_cap_name, top_cap_name, shell_name]
    '''

    # Find the total z-distance travelled
    dz = top - bottom

    # Area for each cap, face, and the total surface area
    Cap_Area = dr**2
    Shell_Face_Area = dr*dz
    Total_Area = 2*Cap_Area + 4*Shell_Face_Area
    
    # Find the number of points for each face (cap or shell)
    num_cap_points = int(round((Cap_Area / Total_Area) * num_points))
    num_face_points = int(round((Shell_Face_Area / Total_Area) * num_points))
    
    # Splitting the grid of points into how many points per row (or column)
    # sqrt should mean that we get the same # of points for each row and column
    # We round up just to make sure we get at least the requested # of points
    num_cap_row_points = int(math.sqrt(num_cap_points) // 1) + 1
    num_shell_row_points = int(math.sqrt(num_face_points) // 1) + 1

    # Size of each step (steps are of same size for the square caps
    r_step = dr / num_cap_row_points
    z_step = dz / num_shell_row_points
    r_step_face = dr / num_shell_row_points
    
    cap_points1 = []
    cap_points2 = []
    # Generate cap points
    for deltax in range(num_cap_row_points + 1):
        for deltay in range(num_cap_row_points + 1):
            point = [-dr/2 + r_step*deltax,
                     -dr/2 + r_step*deltay,
                     bottom]
            cap_points1.append(point)

            point = [-dr/2 + r_step*deltax,
                     -dr/2 + r_step*deltay,
                     top]
            cap_points2.append(point)

    
    shell1 = []
    shell2 = []
    shell3 = []
    shell4 = []
    # Generate shell points
    for i in range(num_shell_row_points + 1):
        for j in range(num_shell_row_points + 1):
            point1 = [-dr/2 + r_step_face * i,
                      -dr/2,
                      bottom + z_step * j]
            shell1.append(point1)

            point2 = [dr/2,
                      -dr/2 + r_step_face * i,
                      bottom + z_step * j]
            shell2.append(point2)

            point3 = [-dr/2 + r_step_face * i,
                      dr/2,
                      bottom + z_step * j]
            shell3.append(point3)

            point4 = [-dr/2,
                      -dr/2 + r_step_face * i,
                      bottom + z_step * j]
            shell4.append(point4)

    full_shell = []
    for i in range(len(shell1)):
        full_shell.append(shell1[i])
        full_shell.append(shell2[i])
        full_shell.append(shell3[i])
        full_shell.append(shell4[i])

    bot_cap = np.array(cap_points1)
    top_cap = np.array(cap_points2)
    shell = np.array(full_shell)
        
        
    np.save(filenames[0], bot_cap)
    np.save(filenames[1], top_cap)
    np.save(filenames[2], shell)
            
    
    
def load_boundary(region=1):
    if region == 1:
        # ALL ARRAYS HERE SHOULD BE OF SHAPE nx6 (COORDINATES + B-FIELD VECTOR)
        top_cap = np.load('UDET Top Cap.npy')
        bot_cap = np.load('UDET Bot Cap.npy')
        shell = np.load('UDET Shell.npy')
        return top_cap, bot_cap, shell

def random_boundary_points(num_points, top_cap, bot_cap, shell, region=1):
    if region == 1:
        radius = 10
        bottom_height = 3
        height = 525 - bottom_height

        cap_ratio = radius / (2*(radius + height))
        shell_ratio = height / (radius + height)


        num_points_caps = int(round(cap_ratio * num_points, 0))
        num_points_shell = int(round(shell_ratio * num_points, 0))

        top_cap_indices = np.random.choice(top_cap.shape[0], size=num_points_caps, replace=False)
        bot_cap_indices = np.random.choice(bot_cap.shape[0], size=num_points_caps, replace=False)
        shell_indices = np.random.choice(shell.shape[0], size=num_points_shell, replace=False)

        return top_cap[top_cap_indices], bot_cap[bot_cap_indices], shell[shell_indices]

#----------------------------------------------------------------------------------------------------------------


#---------------------------ON-AXIS DATA FUNCTIONS-----------------------------------------------------
def load_axis():
    data = np.load('OnAxisOpera.npy')
    return data

def random_axis_points(num_points, data):
    indices = np.random.choice(data.shape[0], size=num_points, replace=False)
    return data[indices]

def generate_axis_points(num_points=65000):
    step = 650/num_points
    z0 = -125
    zf = 525

    points = []
    for i in range(num_points+1):
        point = [0, 0, z0+(i*step)]
        points.append(point)

    points = np.array(points)
    return points
    

#-----------------------PLOTTING-------------------------------------------------------
def plot_axis(coordinates, fields, title=''):
    z_values = coordinates[:,2]
    field_mags = []

    for field in fields:
        mag = np.sqrt(field[0]**2 + field[1]**2 + field[2]**2)
        field_mags.append(mag)

    fig, ax = plt.subplots()
    ax.plot(z_values, field_mags)
    ax.set_xlabel('z (cm)')
    ax.set_ylabel('B (Teslas)')
    ax.set_title(title)
    plt.show()
