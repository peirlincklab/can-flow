import numpy as np
import pandas as pd
import pyvista as pv


x_confounders = pd.read_excel(r"C:\Users\kkevopoulos\OneDrive - Delft University of Technology\Bureaublad\data_kostas_bivme\metadata_final.xlsx")
x_confounders.drop(['Participant ID', 'Height', 'Weight', 'Diastolic BP',
                    'Systolic BP', 'Unnamed: 8', 'Unnamed: 9', 'subject_id'], axis=1, inplace=True)
x_confounders = pd.get_dummies(x_confounders, columns=['Sex'])
x_confounders['Sex_Female'] = x_confounders['Sex_Female'].replace({True: 1, False: 0})
x_confounders['Sex_Male'] = x_confounders['Sex_Male'].replace({True: 1, False: 0})
x_confounders = x_confounders[['BMI', 'Age', 'Sex_Female', 'Sex_Male']]
x_confounders = x_confounders.to_numpy()

### Find the indices of the real subgroup
male_indices = np.where(x_confounders[:, 2] == 0)[0]
x_conf_male = x_confounders[male_indices]

subgroup_indices = np.where(x_conf_male[:, 1] > 58)[0]

### Load all real shapes that belong to this subgroup
input_dir = r"C:\Users\kkevopoulos\Documents\Meshes_Anatomies\Reference_Momenta"

### load the template shape of the cohort
### each node's variance will be visualized on this template
template = pv.read(r"C:\Users\kkevopoulos\Documents\Meshes_Anatomies\template.vtk")


NumAll = 2274
meshes_real = []
for i in range(NumAll):
    if i in subgroup_indices:
        filename = input_dir + f"\Shooting_Momenta_{i}\output\Shooting__GeodesicFlow__biv__tp_10__age_1.00.vtk"
        mesh = pv.read(filename)
        mesh = mesh.points

        meshes_real.append(mesh)

meshes_real = np.array(meshes_real)
meshes_real_std = meshes_real.std(axis=0)

template['std_real'] = meshes_real_std


### For each generative model (nf, vae2, vae3), load all synthetic shapes
models = ['nf', 'vae2', 'vae3']
NumSynth = 650
for model_str in models:

    input_dir_synthetic = rf"C:\Users\kkevopoulos\Documents\Meshes_Anatomies\Targeted_{model_str}_Momenta"
    meshes_synthetic = []
    for j in range(NumSynth):
        filename = input_dir_synthetic + f"\Shooting_Momenta_{j}\output\Shooting__GeodesicFlow__biv__tp_10__age_1.00.vtk"
        mesh = pv.read(filename)
        mesh = mesh.points

        meshes_synthetic.append(mesh)

    meshes_synthetic = np.array(meshes_synthetic)
    meshes_synthetic_std = meshes_synthetic.std(axis=0)

    template[f'std_{model_str}'] = meshes_synthetic_std

template.save(f"../data_models_saved/data/template_with_variances.vtk")

