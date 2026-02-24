import pyvista as pv
import numpy as np


individual = 2264

directory = r"C:\Users\kkevopoulos\Documents\Meshes_Anatomies\Reference_Momenta"
filename = directory + f"\Shooting_Momenta_{individual}\output\Shooting__GeodesicFlow__biv__tp_10__age_1.00.vtk"

real_individual = pv.read(filename)

num_gen = 70
meshes_synthetic = []
for i in range(num_gen):
    dir_synth = r"C:\Users\kkevopoulos\Documents\Meshes_Anatomies\Specific_individual_Momenta"
    filename_synth = dir_synth + f"\Shooting_Momenta_{i}\output\Shooting__GeodesicFlow__biv__tp_10__age_1.00.vtk"

    mesh = pv.read(filename_synth)
    mesh = mesh.points

    meshes_synthetic.append(mesh)

meshes_synthetic = np.array(meshes_synthetic)
meshes_synthetic_std = meshes_synthetic.std(axis=0)

real_individual['std'] = meshes_synthetic_std
real_individual.save(f"../data_models_saved/data/real_individual_std.vtk")
