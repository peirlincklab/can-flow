import pyvista as pv
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
import os
from scipy.spatial.distance import directed_hausdorff


def mean_surface_distance(points1, points2):
    """Compute the mean surface distance between two sets of points."""
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)

    distances1, _ = tree1.query(points2)  # Distance from mesh2 to mesh1
    distances2, _ = tree2.query(points1)  # Distance from mesh1 to mesh2

    return (np.mean(distances1) + np.mean(distances2)) / 2


def hausdorff_distance(points1, points2):
    d_AB = directed_hausdorff(points1, points2)[0]  # A to B
    d_BA = directed_hausdorff(points2, points1)[0]  # B to A

    # Get symmetric Hausdorff distance
    hausdorff_dist = max(d_AB, d_BA)

    return hausdorff_dist


def load_vtk_mesh(file_path):
    """Load a surface mesh from a VTK file using PyVista and return its points."""
    mesh = pv.read(file_path)
    return np.array(mesh.points)


def performance_distances(custom_range, distance_func):

    directory = r"C:\Users\kkevopoulos\Documents"

    structures = ['\Reference_Momenta', '\Predicted_Momenta']
    meshes_ref = []
    meshes_pred = []

    dist_values = []

    for struct in structures:
        directory_struct = directory + struct

        pbar = tqdm(total=custom_range, desc="loading meshes...")
        for i in range(custom_range):

            subfolders_path = f"Shooting_Momenta_{i+433}\output\Shooting__GeodesicFlow__biv__tp_10__age_1.00.vtk"
            full_path = os.path.join(directory_struct, subfolders_path)

            mesh = load_vtk_mesh(full_path)

            if struct == '\Reference_Momenta':
                meshes_ref.append(mesh)
            else:
                meshes_pred.append(mesh)

            pbar.update()
        pbar.close()


    pbar = tqdm(total=custom_range, desc="computing distances...")
    for j in range(custom_range):
        dist = distance_func(meshes_ref[j], meshes_pred[j])
        dist_values.append(dist)
        pbar.update()
    pbar.close()

    mean_dist = np.mean(dist_values)
    print(f"Mean value of {str(distance_func)} over samples = {mean_dist} +- {np.std(dist_values)}")

    return mean_dist

performance_distances(23, hausdorff_distance)



