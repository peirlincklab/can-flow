import os
import shutil
from tqdm import tqdm



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

num_gen = 650 ### 650 synthetic anatomies for each model

models = ['nf', 'vae2', 'vae3']
for model in models:
    copy2clean(model_str=model, all_num=num_gen, model_str_path=f"Targeted_{model}_Momenta")



