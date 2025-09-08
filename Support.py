import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
projects_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ferenc_path = os.path.join(projects_dir, "Nab Magnetometry/Bferenc Files/Magnetometry-master/Bferenc Python/")
sys.path.append(ferenc_path)
magfield_path = os.path.join(ferenc_path, "magfield")
sys.path.append(magfield_path)

from MainRoutine import mainmagfield
import magfield


#-----------------------------BOUNDARY DATA FUNCTIONS------------------------------------------------------------
def get_ratios():
    # Geometric constants
    r_top = 10
    r_mid = 2
    z_top = 500
    z_1 = 1.5
    z_2 = -1.5
    z_bot = -100

    # Area calculations
    A_topcap = np.pi * r_top**2
    A_botcap = np.pi * r_top**2
    A_topwall = 2 * np.pi * r_top * (z_top - z_1)
    A_botwall = 2 * np.pi * r_top * (z_2 - z_bot)
    A_midwall = 2 * np.pi * r_mid * (z_1 - z_2)
    A_up_ring = np.pi * (r_top**2 - r_mid**2)
    A_lo_ring = np.pi * (r_top**2 - r_mid**2)

    areas = np.array([A_topcap, A_botcap, A_topwall, A_botwall, A_midwall, A_up_ring, A_lo_ring])

    # Fraction of points per surface
    fractions = areas / areas.sum()
    return fractions

def get_surface_points(boundary_pts, N):
    fractions = get_ratios()
    counts = np.round(fractions * N).astype(int)

    # Ensure sum exactly equals N
    while counts.sum() < N:
        counts[np.argmax(fractions)] += 1
    while counts.sum() > N:
        counts[np.argmax(counts)] -= 1

    surfaces = ["topcap", "botcap", "topshell", "botshell", "midshell", "upring", "lowring"]

    points = []
    # Pull a total of N points from the boundaries
    for i, surface in enumerate(surfaces):
        n = counts[i]
        indices = np.random.choice(boundary_pts[surface].shape[0], size=n, replace=False)
        npoints = boundary_pts[surface][indices, :]
        for point in npoints:
            points.append(point)
    
    points = np.array(points)
    return points



def load_boundary_points():
    points = {}

    # Directory and file names of the different boundary parts to load training data from
    folder_dir = "S:/Python/Projects/Magnetometry-Neural-Networks/boundary_points"
    files = ["topcap", "botcap", "topshell", "botshell", "midshell", "upring", "lowring"]

    # Load the precalculated points and pull n random points from them.
    for file in files:
        points[file] = np.load(folder_dir+"/"+file+".npy")
    
    
    return points


def generate_random_points_in_volume(N):
    # Cylinder definitions
    # (zmin, zmax, radius)
    regions = [
        (1.5, 500, 10),     # Top
        (-1.5, 1.5, 2),     # Middle
        (-100, -1.5, 10)    # Bottom
    ]
    
    # Calculate volumes for probability weighting
    volumes = []
    for zmin, zmax, r in regions:
        h = zmax - zmin
        volumes.append(np.pi * r**2 * h)
    volumes = np.array(volumes)
    probs = volumes / volumes.sum()
    
    # Decide which region each point belongs to
    region_choices = np.random.choice(len(regions), size=N, p=probs)
    
    # Pre-allocate
    points = np.zeros((N, 3))
    
    for idx, region_idx in enumerate(region_choices):
        zmin, zmax, rmax = regions[region_idx]
        # Uniform z in the region
        z = np.random.uniform(zmin, zmax)
        # Uniformly in a disk (not clustered at center)
        theta = np.random.uniform(0, 2*np.pi)
        rr = rmax * np.sqrt(np.random.uniform(0,1))
        x = rr * np.cos(theta)
        y = rr * np.sin(theta)
        points[idx] = [x, y, z]
        
    return points
  
    


#---------------------INTERNAL POINT SAMPLER-----------------------------------------------------------
def get_internal_points(N):
    # Geometric constants
    r_top = 10
    r_mid = 2
    z_top = 500
    z_1 = 1.5
    z_2 = -1.5
    z_bot = -100

    # Volumes (π R^2 h)
    top_vol = np.pi * (r_top**2) * (z_top - z1)
    mid_vol = np.pi * (r_mid**2) * (z1 - z2)
    bot_vol = np.pi * (r_top**2) * (z2 - z_bot)

    vols = np.array([top_vol, mid_vol, bot_vol], dtype=float)
    fracs = vols / vols.sum()

    # Allocate counts with deterministic rounding that sums to N
    counts = np.floor(fracs * N).astype(int)

    n_top, n_mid, n_bot = counts.tolist()

    def sample_cylinder(n, r_max, z_min, z_max):
        if n <= 0:
            return np.empty((0, 3))
        # Uniform in height and angle; r via sqrt trick for area-uniform disks
        z = rng.uniform(z_min, z_max, size=n)
        theta = rng.uniform(0.0, 2.0*np.pi, size=n)
        r = r_max * np.sqrt(rng.uniform(0.0, 1.0, size=n))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack((x, y, z))

    # Sample each region
    top_pts = sample_cylinder(n_top, r_top, z1, z_top)
    mid_pts = sample_cylinder(n_mid, r_mid, z2, z1)
    bot_pts = sample_cylinder(n_bot, r_top, z_bot, z2)

    pts = np.vstack((top_pts, mid_pts, bot_pts))
    np.random.default_rng().shuffle(pts)  # de-bias any ordering
    return pts    

#--------------------------FERENC ROUTINE---------------------------------------
def get_field(points):
    # Convert points to m
    points = points/100
    B = mainmagfield(points=points, asfunction=True)
    return B






def save_surface_points(
    N_total=1000000, # Total points per surface
    seed=None,
    out_dir="boundary_points"
):

    if seed is not None:
        np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # Geometric constants
    r_top = 10
    r_mid = 2
    z_top = 500
    z_1 = 1.5
    z_2 = -1.5
    z_bot = -100

    # 1. Top cap: z = z_top, r in [0, r_top]
    n = N_total
    r = r_top * np.sqrt(np.random.rand(n))
    theta = 2 * np.pi * np.random.rand(n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.full(n, z_top)
    points = np.column_stack([x, y, z])
    ferenc_points = points/100
    field = mainmagfield(points=ferenc_points, asfunction=True)
    np.savetxt(f"{out_dir}/topcap_points.txt", np.hstack((points, field)), fmt="%.6f")

    # 2. Bottom cap: z = z_bot, r in [0, r_top]
    r = r_top * np.sqrt(np.random.rand(n))
    theta = 2 * np.pi * np.random.rand(n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.full(n, z_bot)
    points = np.column_stack([x, y, z])
    ferenc_points = points/100
    field = mainmagfield(points=ferenc_points, asfunction=True)
    np.savetxt(f"{out_dir}/botcap_points.txt", np.hstack((points, field)), fmt="%.6f")

    # 3. Top wall: r = r_top, z in (z_1, z_top]
    z = np.random.uniform(z_1, z_top, n)
    theta = 2 * np.pi * np.random.rand(n)
    x = r_top * np.cos(theta)
    y = r_top * np.sin(theta)
    points = np.column_stack([x, y, z])
    ferenc_points = points/100
    field = mainmagfield(points=ferenc_points, asfunction=True)
    np.savetxt(f"{out_dir}/topwall_points.txt", np.column_stack([x, y, z]), fmt="%.6f")

    # 4. Bottom wall: r = r_top, z in [z_bot, z_2)
    z = np.random.uniform(z_bot, z_2, n)
    theta = 2 * np.pi * np.random.rand(n)
    x = r_top * np.cos(theta)
    y = r_top * np.sin(theta)
    points = np.column_stack([x, y, z])
    ferenc_points = points/100
    field = mainmagfield(points=ferenc_points, asfunction=True)
    np.savetxt(f"{out_dir}/botwall_points.txt", np.column_stack([x, y, z]), fmt="%.6f")

    # 5. Middle wall: r = r_mid, z in [z_2, z_1]
    z = np.random.uniform(z_2, z_1, n)
    theta = 2 * np.pi * np.random.rand(n)
    x = r_mid * np.cos(theta)
    y = r_mid * np.sin(theta)
    points = np.column_stack([x, y, z])
    ferenc_points = points/100
    field = mainmagfield(points=ferenc_points, asfunction=True)
    np.savetxt(f"{out_dir}/midwall_points.txt", np.column_stack([x, y, z]), fmt="%.6f")

    # 6. Upper ring: z = z_1, r in (r_mid, r_top]
    r = np.sqrt(np.random.uniform(r_mid**2, r_top**2, n))
    theta = 2 * np.pi * np.random.rand(n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.full(n, z_1)
    points = np.column_stack([x, y, z])
    ferenc_points = points/100
    field = mainmagfield(points=ferenc_points, asfunction=True)
    np.savetxt(f"{out_dir}/up_ring_points.txt", np.column_stack([x, y, z]), fmt="%.6f")

    # 7. Lower ring: z = z_2, r in (r_mid, r_top]
    r = np.sqrt(np.random.uniform(r_mid**2, r_top**2, n))
    theta = 2 * np.pi * np.random.rand(n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.full(n, z_2)
    points = np.column_stack([x, y, z])
    ferenc_points = points/100
    field = mainmagfield(points=ferenc_points, asfunction=True)
    np.savetxt(f"{out_dir}/lo_ring_points.txt", np.column_stack([x, y, z]), fmt="%.6f")

    print(f"Saved {N_total} points for each surface in '{out_dir}/'.")






def convert_txt_to_npy(directory):
    """
    Converts all .txt files in a directory to .npy files (same base filename).
    Assumes each txt file contains plain whitespace-separated columns.
    """
    for fname in os.listdir(directory):
        if fname.endswith('.txt'):
            txt_path = os.path.join(directory, fname)
            npy_path = os.path.join(directory, fname.replace('.txt', '.npy'))
            print(f"Converting {txt_path} to {npy_path} ...")
            arr = np.loadtxt(txt_path)
            np.save(npy_path, arr)
    print("All txt files converted to npy.")




#---------------------------ON-AXIS DATA FUNCTIONS-----------------------------------------------------
def load_axis():
    data = np.load('OnAxisOpera.npy')
    return data

def random_axis_points(num_points, data):
    indices = np.random.choice(data.shape[0], size=num_points, replace=False)
    return data[indices]

def generate_axis_points(num_points=60000):
    step = 600/num_points
    z0 = -100
    zf = 500

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



from mpl_toolkits.mplot3d import Axes3D

def plot_boundary_points(points, title="Boundary Points",
                         xstart=None, xend=None, ystart=None, yend=None,
                         zstart=None, zend=None):
    """
    Plots 3D boundary points.

    Args:
        points (np.ndarray): Array of shape (n, 3) containing x, y, z coordinates of points.
        title (str): Title for the plot.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.6)
    
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    if xstart is not None and xend is not None:
        plt.xlim(xstart, xend)
    if ystart is not None and yend is not None:
        plt.ylim(ystart, yend)
    if zstart is not None and zend is not None:
        ax.set_zlim(zstart, zend)
    ax.set_title(title)
    plt.show()


if __name__ == "__main__":
    files = ["topshell", "botshell", "midshell", "upring", "lowring"]
    for file in files:
        data = np.load(file+".npy")
        print(data.shape)
    sys.exit()
    
    folder_dir = "S:/Python/Projects/Magnetometry-Neural-Networks/boundary_points"
    files = ["topcap", "botcap", "topshell", "botshell", "midshell", "upring", "lowring"]

    for file in files:
        data = np.load(folder_dir+"/"+file+".npy")
        
        if data.shape[1] != 6:
            print(f"Calculating field for {file}...")
            B = get_field(data)

            print(f"Writing {file}...")
            with open(file+".txt", 'w') as f:
                counter = 0
                for i, field in enumerate(B):
                    f.write(f"{data[i,0]}\t{data[i,1]}\t{data[i,2]}\t{field[0]}\t{field[1]}\t{field[2]}\n")


