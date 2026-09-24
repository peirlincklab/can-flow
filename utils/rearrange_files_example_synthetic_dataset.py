import os
import shutil
from tqdm import tqdm



def copy2clean(model_str, all_num, model_str_path):

    os.makedirs(fr"C:\Users\kkevopoulos\Documents\Meshes_Anatomies_Alternative_Branch\Gen_Metrics_Exp_Files\{model_str}_Meshes", exist_ok=True)

    pbar = tqdm(total=all_num, desc='Copy PCs to clean folder...')
    for i in range(all_num):

        vtk_src_path = fr"C:\Users\kkevopoulos\Documents\Meshes_Anatomies_Alternative_Branch\{model_str_path}\Shooting_Momenta_{i}\output\Shooting__GeodesicFlow__biv__tp_10__age_1.00.vtk"
        new_folder_path =  fr"C:\Users\kkevopoulos\Documents\Meshes_Anatomies_Alternative_Branch\Gen_Metrics_Exp_Files\{model_str}_Meshes/Mesh_{i}.vtk"

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




num_all = 20

sex_female = [1, 0]
sex_male = [0, 1]

age1 = [50, 60]
age2 = [60, 70]
age3 = [70, 80]

bmi1 = [17, 21]
bmi2 = [21, 25]
bmi3 = [25, 29]

sex_vals = [sex_female, sex_male]
age_vals = [age1, age2, age3]
bmi_vals = [bmi1, bmi2, bmi3]


for sex in sex_vals:
    for age in age_vals:
        for bmi in bmi_vals:
            sex_str = 'female' if sex == [1, 0] else 'male'

            copy2clean(model_str=f'{sex_str}_{age}_{bmi}', all_num=num_all, model_str_path=fr"Synthetic_Example_Dataset\{sex_str}_{age}_{bmi}_Momenta")




