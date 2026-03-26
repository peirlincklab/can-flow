import numpy as np
from tqdm import tqdm
import pyvista as pv


template = pv.read(r"C:\Users\kkevopoulos\Documents\Meshes_Anatomies\template.vtk")
input_dir = r"C:\Users\kkevopoulos\Documents\Meshes_Anatomies\Reference_Reconstructed_Momenta"


NumAll = 2208
meshes_real = []
pbar = tqdm(total=NumAll, desc='Loading anatomies...', leave=True)
for i in range(NumAll):
    filename = input_dir + f"\Shooting_Momenta_{i}\output\Shooting__GeodesicFlow__biv__tp_10__age_1.00.vtk"
    mesh = pv.read(filename)
    mesh = mesh.points

    meshes_real.append(mesh)

    pbar.update()
pbar.close()

meshes_real = np.array(meshes_real)
meshes_real_std = meshes_real.std(axis=0)

template['std_real'] = meshes_real_std
template.save(f"template_with_variances_ae.vtk")