import os
import shutil
from tqdm import tqdm
import pandas as pd
import numpy as np



def copy2clean(model_str, all_num, model_str_path):

    os.makedirs(fr"C:\Users\kkevopoulos\Documents\Meshes_Anatomies\Gen_Metrics_Exp_Files_Targeted\{model_str}_PCs", exist_ok=True)

    pbar = tqdm(total=all_num, desc='Copy PCs to clean folder...')
    for i in range(all_num):

        vtk_src_path = fr"C:\Users\kkevopoulos\Documents\Meshes_Anatomies\{model_str_path}\Shooting_Momenta_{i}\output\Shooting__GeodesicFlow__biv__tp_10__age_1.00.vtk"
        new_folder_path =  fr"C:\Users\kkevopoulos\Documents\Meshes_Anatomies\Gen_Metrics_Exp_Files_Targeted\{model_str}_PCs\PointCloud_{i}.vtk"

        # Ensure source exists
        if not os.path.exists(vtk_src_path):
            print("WARNING: source missing:", vtk_src_path)
            pbar.update()
            continue

        try:
            shutil.copy(src=vtk_src_path, dst=new_folder_path)
        except PermissionError as e:
            print("Permission error copying to", new_folder_path, "-", e)
        except Exception as e:
            print("Copy failed:", e)

        pbar.update()
    pbar.close()



os.makedirs( r"C:\Users\kkevopoulos\Documents\Meshes_Anatomies\Gen_Metrics_Exp_Files_Targeted", exist_ok=True)

### real patients of the subgroup
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

os.makedirs(fr"C:\Users\kkevopoulos\Documents\Meshes_Anatomies\Gen_Metrics_Exp_Files_Targeted\Real_PCs", exist_ok=True)

NumAll = 2274
j=0
pbar = tqdm(total=NumAll, desc='Copy PCs to clean folder...')
for i in range(NumAll):
    if i in subgroup_indices:

        vtk_src_path = fr"C:\Users\kkevopoulos\Documents\Meshes_Anatomies\Reference_Momenta\Shooting_Momenta_{i}\output\Shooting__GeodesicFlow__biv__tp_10__age_1.00.vtk"
        new_folder_path = fr"C:\Users\kkevopoulos\Documents\Meshes_Anatomies\Gen_Metrics_Exp_Files_Targeted\Real_PCs\PointCloud_{j}.vtk"

        # Ensure source exists
        if not os.path.exists(vtk_src_path):
            print("WARNING: source missing:", vtk_src_path)
            pbar.update()
            continue

        try:
            shutil.copy(src=vtk_src_path, dst=new_folder_path)
        except PermissionError as e:
            print("Permission error copying to", new_folder_path, "-", e)
        except Exception as e:
            print("Copy failed:", e)
        j += 1

    pbar.update()
pbar.close()



num_gen = 650 ### 650 synthetic anatomies for each model

models = ['nf', 'vae2', 'vae3']
for model in models:
    copy2clean(model_str=model, all_num=num_gen, model_str_path=f"Targeted_{model}_Momenta")



