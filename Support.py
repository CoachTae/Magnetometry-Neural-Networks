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
        bottom_height = 1.5
        height = 525 - bottom_height
    
    
    elif region == 2: # F coil region
        radius = 4.5
        bottom_height = -1.5
        height = 1.5 - bottom_height
    
    
    elif region == 3: # LDET region
        radius = 10
        bottom_height = -125
        height = -1.5 - bottom_height

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

    sqrt(2)/2 * max_radius should give you a good dr for rectangular boxes.
        e.g. F coil has an inner radius of 4.57cm (let's round to 4.5 to be safe
             4.5 * sqrt(2)/2 = (the dr we should use for this).
             To see this, draw a square of side lengths dr
                 draw the diagonal (longest line along the square)
                 this diagonal is our max distance (4.5)
                 any longer and we'd hit our coil
                 since it's a square, it's an equilateral triangle
                 dr/4.5 = cos(45)
                 dr = 4.5 * sqrt(2)/2

    dr gives the total width along x and y.
        e.g. dr = 20 means x spans from -10 to +10, and y does the same

    bottom is the z-coordinate corresponding to the bottom of the box

    top is the z-coordinate corresponding to the top of the box

    num_points is the number of points for the entire surface

    filenames should be a list like [bot_cap_name, top_cap_name, shell_name]

    Gives coordinates in .table form for Opera to calculate values at
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
        
        
    npy_to_table(bot_cap, filenames[0])
    npy_to_table(top_cap, filenames[1])
    npy_to_table(shell, filenames[2])
            
    
    
def load_boundary(region):
    if region == 1:
        # ALL ARRAYS HERE SHOULD BE OF SHAPE nx6 (COORDINATES + B-FIELD VECTOR)
        top_cap = np.load('UDET_Top_Cap.npy')
        bot_cap = np.load('UDET_Bot_Cap.npy')
        shell = np.load('UDET_Shell.npy')
        return top_cap, bot_cap, shell

    elif region == 2:
        top_cap = np.load('F_Top_Cap.npy')
        bot_cap = np.load('F_Bot_Cap.npy')
        shell = np.load('F_Shell.npy')
        return top_cap, bot_cap, shell

    elif region == 3:
        top_cap = np.load('LDET_Top_Cap.npy')
        bot_cap = np.load('LDET_Bot_Cap.npy')
        shell = np.load('LDET_Shell.npy')
        return top_cap, bot_cap, shell

def random_boundary_points(num_points, top_cap, bot_cap, shell, region):
    if region == 1:
        width = 10
        bottom_height = 1.5
        height = 550 - bottom_height

        

    elif region == 2:
        width = 3.18
        bottom_height = -1.5
        height = 1.5 - bottom_height


    elif region == 3:
        width = 10
        bottom_height = -150
        height = -1.5 - bottom_height

        
    cap_area = width**2
    face_area = width*height
    total_area = 2*cap_area + 4*face_area

    cap_ratio = cap_area / total_area
    shell_ratio = 4*face_area / total_area


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

def generate_axis_points(num_points=70000):
    step = 700/num_points
    z0 = -150
    zf = 550

    points = []
    for i in range(num_points+1):
        point = [0, 0, z0+(i*step)]
        points.append(point)

    points = np.array(points)
    return points
    

#-----------------------PLOTTING-------------------------------------------------------
def plot_axis(coordinates, fields, scatter=False, title=''):
    z_values = coordinates[:,2]
    field_mags = []

    for field in fields:
        mag = np.sqrt(field[0]**2 + field[1]**2 + field[2]**2)
        field_mags.append(mag)

    fig, ax = plt.subplots()
    if not scatter:
        ax.plot(z_values, field_mags)
    else:
        ax.scatter(z_values, field_mags, s=10)
    ax.set_xlabel('z (cm)')
    ax.set_ylabel('B (Teslas)')
    ax.set_title(title)
    plt.show()



#-------------------OPERA FUNCTIONS----------------------------------------------

def npy_to_table(array, filename):
    with open(filename, 'w') as file:
        file.write(str(len(array[:,0])) + ' 1 1 2\n')
        file.write('1 X [CM]\n')
        file.write('2 Y [CM]\n')
        file.write('3 Z [CM]\n')
        file.write('0\n')

        for point in array:
            file.write(f'{point[0]}\t{point[1]}\t{point[2]}\n')

def table_to_npy(files):
    '''
    Files can be a single file or a list/tuple of files

    EXPECTS UNITS OF T, NOT GAUSS
    '''
    def write_array(filename):
        with open(filename, 'r') as file:
            # Skip lines until we finish the header
            while True:
                line = file.readline().strip()
                if line == '0':
                    break
            data = np.loadtxt(file)
            output_file = filename.replace('table', 'npy')
            np.save(output_file, data)

    if isinstance(files, str):
        write_array(files)

    elif isinstance(files, list) or isinstance(files, tuple):
        for file in files:
            write_array(file)
    else:
        print("Error in filename for table_to_npy.")
        print(f"Type given: {type(files)}")

