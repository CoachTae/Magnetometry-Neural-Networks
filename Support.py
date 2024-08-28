import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

# Region 1 will be anything above the filter region
# Region 2 will be the filter region, very small
# Region 3 will be anything below the filter region
# This function is for use in the neural network's interior scan to ensure that any given point within a region obeys Maxwell's Equations
def random_points(num_points, device='cuda'):
    # To be redone after removal of references to "region" is complete
    pass
        


#-----------------------------BOUNDARY DATA FUNCTIONS------------------------------------------------------------
def generate_boundary_coordinates(step, filenames):
    '''
    Given a step size between points, generate points equally
        distributed among the boundary specified.

    This function will need to be tailored to any given boundary shape.

    The resulting points need to be run through a simulation software.
    In my case, I'm processing the field at these points through Opera3D.
    '''

    def get_points(start, end, step):
        '''
        Define a rectangle with 2 points.

        Generate equidistant points on that surface.
        '''

        points = []

        k = 0
        while k*step <= end[2] - start[2]:
            j=0
            while j*step <= end[1] - start[1]:
                i=0
                while i*step <= end[0] - start[0]:
                    points.append([round(start[0] + i*step, 5),
                                   round(start[1] + j*step, 5),
                                   round(start[2] + k*step, 5)])
                    i += 1
                j += 1
            k += 1

        return points

    # UDET Cap
    start = (-10, -10, 510)
    end = (10, 10, 510)
    top_cap = get_points(start, end, step)
    top_cap = np.array(top_cap)

    # UDET Shell
    start = (-10, -10, 1.5)
    end = (10, -10, 510)
    UDET_Shell1 = get_points(start, end, step)
    UDET_Shell1 = np.array(UDET_Shell1)

    start = (10, -10, 1.5)
    end = (10, 10, 510)
    UDET_Shell2 = get_points(start, end, step)
    UDET_Shell2 = np.array(UDET_Shell2)

    start = (-10, 10, 1.5)
    end = (10, 10, 510)
    UDET_Shell3 = get_points(start, end, step)
    UDET_Shell3 = np.array(UDET_Shell3)

    start = (-10, -10, 1.5)
    end = (-10, 10, 510)
    UDET_Shell4 = get_points(start, end, step)
    UDET_Shell4 = np.array(UDET_Shell4)

    

    # The bottom cap of the UDET box
    # We need to remove the parts that are inside to F region shell
    start = (-10, -10, 1.5)
    end = (10, 10, 1.5)
    UDET_Bottom = get_points(start, end, step)
    UDET_Bottom = np.array(UDET_Bottom)

    # If x is within the confines of the F shell
    condition1 = (-3.182 < UDET_Bottom[:,0]) & (UDET_Bottom[:,0] < 3.182)

    # IF y is within the confines of the F shell
    condition2 = (-3.182 < UDET_Bottom[:,1]) & (UDET_Bottom[:,1] < 3.182)

    combined_condition = condition1 & condition2

    # Mark the indices in need of deletion
    indices = np.where(combined_condition)[0]

    UDET_Bottom = np.delete(UDET_Bottom, indices, axis=0)



    # F region shell
    start = (-3.182, -3.182, -1.5)
    end = (3.182, -3.182, 1.5)
    F_Shell1 = get_points(start, end, step)
    F_Shell1 = np.array(F_Shell1)

    start = (3.182, -3.182, -1.5)
    end = (3.182 ,3.182, 1.5)
    F_Shell2 = get_points(start, end, step)
    F_Shell2 = np.array(F_Shell2)

    start = (-3.182, 3.182, -1.5)
    end = (3.182, 3.182, 1.5)
    F_Shell3 = get_points(start, end, step)
    F_Shell3 = np.array(F_Shell3)

    start = (-3.182, -3.182, -1.5)
    end = (-3.182, 3.182, 1.5)
    F_Shell4 = get_points(start, end, step)
    F_Shell4 = np.array(F_Shell4)



    # LDET Top cap with the inner region inside the F Shell taken out
    start = (-10, -10, -1.5)
    end = (10, 10, -1.5)
    LDET_Top = get_points(start, end, step)
    LDET_Top = np.array(LDET_Top)

    # If x is within the confines of the F shell
    condition1 = (-3.182 < LDET_Top[:,0]) & (LDET_Top[:,0] < 3.182)

    # IF y is within the confines of the F shell
    condition2 = (-3.182 < LDET_Top[:,1]) & (LDET_Top[:,1] < 3.182)

    combined_condition = condition1 & condition2

    # Mark the indices in need of deletion
    indices = np.where(combined_condition)[0]

    LDET_Top = np.delete(LDET_Top, indices, axis=0)


    # LDET Shell
    start = (-10, -10, -125)
    end = (10, -10, -1.5)
    LDET_Shell1 = get_points(start, end, step)
    LDET_Shell1 = np.array(LDET_Shell1)

    start = (10, -10, -125)
    end = (10, 10, -1.5)
    LDET_Shell2 = get_points(start, end, step)
    LDET_Shell2 = np.array(LDET_Shell2)

    start = (-10, 10, -125)
    end = (10, 10, -1.5)
    LDET_Shell3 = get_points(start, end, step)
    LDET_Shell3 = np.array(LDET_Shell3)

    start = (-10, -10, -125)
    end = (-10, 10, -1.5)
    LDET_Shell4 = get_points(start, end, step)
    LDET_Shell4 = np.array(LDET_Shell4)


    # LDET Cap
    start = (-10, -10, -125)
    end = (10, 10, -125)
    bot_cap = get_points(start, end, step)
    bot_cap = np.array(bot_cap)

    all_arrays = (top_cap,
                  UDET_Shell1,
                  UDET_Shell2,
                  UDET_Shell3,
                  UDET_Shell4,
                  UDET_Bottom,
                  F_Shell1,
                  F_Shell2,
                  F_Shell3,
                  F_Shell4,
                  LDET_Top,
                  LDET_Shell1,
                  LDET_Shell2,
                  LDET_Shell3,
                  LDET_Shell4,
                  bot_cap)

    print("Top Cap: ", top_cap.shape[0], " Points")
    print("UDET Shell: ", UDET_Shell1.shape[0], " Points per face")
    print("UDET Bottom: ", UDET_Bottom.shape[0], " Points")
    print("F Shell: ", F_Shell1.shape[0], " Points per face")
    print("LDET Top: ", LDET_Top.shape[0], " Points")
    print("LDET Shell: ", LDET_Shell1.shape[0], " Points per face")
    print("Bot Cap: ", bot_cap.shape[0], " Points")

    all_points = np.vstack(all_arrays)

    print("All points: ", all_points.shape[0])

    np.save(filename, all_points)
    


def random_boundary_points(num_points, top_cap, bot_cap, shell, region):
   # Not sure if needed but might have to redo this during rework
   pass

   



    

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

