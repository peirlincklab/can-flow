import pyvista as pv
import numpy as np
from tqdm import tqdm

models = ['vae1', 'vae2', 'vae3', 'vae4', 'vae5', 'vae6']
settings_z = ['fixed', 'varied']
NumSynth = 300

template = pv.read(r"C:\Users\kkevopoulos\Documents\Meshes_Anatomies\template.vtk")

std_settings = []
for setting in settings_z:

    meshes_models_std = []
    pbar = tqdm(total=len(models), desc='Compute node variability for models...', leave=True)
    for model_str in models:
        input_dir_synthetic = rf"C:\Users\kkevopoulos\Documents\Meshes_Anatomies\VAE_Variability_z_{setting}_{model_str}_Momenta"

        meshes_synthetic = []
        for j in range(NumSynth):
            filename = input_dir_synthetic + f"\Shooting_Momenta_{j}\output\Shooting__GeodesicFlow__biv__tp_10__age_1.00.vtk"
            mesh = pv.read(filename)
            mesh = mesh.points

            meshes_synthetic.append(mesh)

        meshes_synthetic = np.array(meshes_synthetic)
        meshes_synthetic_std = meshes_synthetic.std(axis=0)

        meshes_models_std.append(meshes_synthetic_std)
        pbar.update()
    pbar.close()

    std_settings.append(meshes_models_std)


variability_ratios = [std_settings[0][i] / std_settings[1][i] for i in range(len(models))]

for i, model_str in enumerate(models):
    template[f'variab_ratio_{model_str}'] = variability_ratios[i]

template.save(f"../data_models_saved/data/template_with_variability_ratios.vtk")