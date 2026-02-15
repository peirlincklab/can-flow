import os
import shutil
from tqdm import tqdm



def copy2clean(model_str, all_num, model_str_path, sex=None):

    os.makedirs(fr"C:\Users\kkevopoulos\Documents\Meshes_Anatomies\Gen_Metrics_Exp_Files\{model_str}_PCs", exist_ok=True)

    pbar = tqdm(total=all_num, desc='Copy PCs to clean folder...')
    for i in range(all_num):

        vtk_src_path = fr"C:\Users\kkevopoulos\Documents\Meshes_Anatomies\{model_str_path}\Shooting_Momenta_{i}\output\Shooting__GeodesicFlow__biv__tp_10__age_1.00.vtk"

        if sex is not None:
            new_folder_path =  fr"C:\Users\kkevopoulos\Documents\Meshes_Anatomies\Gen_Metrics_Exp_Files\{model_str}_PCs/PointCloud_{i}_{sex}.vtk"
        else:
            new_folder_path =  fr"C:\Users\kkevopoulos\Documents\Meshes_Anatomies\Gen_Metrics_Exp_Files\{model_str}_PCs/PointCloud_{i}.vtk"

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



os.makedirs( r"C:\Users\kkevopoulos\Documents\Meshes_Anatomies\Gen_Metrics_Exp_Files", exist_ok=True)

num_all_real = 2274
num_gen = 300 ### 300 female and 300 male synthetic anatomies for each model

### copy the real pcs
copy2clean(model_str='Real', all_num=num_all_real, model_str_path="Reference_Momenta")

### nf
copy2clean(model_str='nf', all_num=num_gen, model_str_path="Female_gen_nf_Momenta", sex="female")
copy2clean(model_str='nf', all_num=num_gen, model_str_path="Male_gen_nf_Momenta", sex="male")

### vaes
vaes = ['vae1', 'vae2', 'vae3', 'vae4', 'vae5', 'vae6']
for vae_model in vaes:
    copy2clean(model_str=vae_model, all_num=num_gen, model_str_path=f"Female_gen_{vae_model}_Momenta", sex="female")
    copy2clean(model_str=vae_model, all_num=num_gen, model_str_path=f"Male_gen_{vae_model}_Momenta", sex="male")



