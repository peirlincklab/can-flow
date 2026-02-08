import os
import shutil
from tqdm import tqdm



def copy2clean(model_str, all_num, model_str_path, sex=None):
    if model_str == 'nf':
        folder_name = 'cNF'
    elif model_str == 'vae':
        folder_name = 'cVAE'
    elif model_str == 'gan':
        folder_name = 'cGAN'
    else:
        folder_name = 'Real'

    os.makedirs(fr"C:\Users\kkevopoulos\Documents\Meshes_Anatomies\Gen_Metrics_Exp_Files\{folder_name}_PCs", exist_ok=True)

    pbar = tqdm(total=all_num, desc='Copy PCs to clean folder...')
    for i in range(all_num):

        vtk_src_path = fr"C:\Users\kkevopoulos\Documents\Meshes_Anatomies\{model_str_path}\Shooting_Momenta_{i}\output\Shooting__GeodesicFlow__biv__tp_10__age_1.00.vtk"

        if sex is not None:
            new_folder_path =  fr"C:\Users\kkevopoulos\Documents\Meshes_Anatomies\Gen_Metrics_Exp_Files\{folder_name}_PCs/PointCloud_{i}_{sex}.vtk"
        else:
            new_folder_path =  fr"C:\Users\kkevopoulos\Documents\Meshes_Anatomies\Gen_Metrics_Exp_Files\{folder_name}_PCs/PointCloud_{i}.vtk"

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
num_gen = 700

### copy the real pcs
copy2clean(model_str='Real', all_num=num_all_real, model_str_path="Reference_Momenta")

### nf
copy2clean(model_str='nf', all_num=num_gen, model_str_path="Female_gen_nf_Momenta", sex="female")
copy2clean(model_str='nf', all_num=num_gen, model_str_path="Male_gen_nf_Momenta", sex="male")

### vae
copy2clean(model_str='vae', all_num=num_gen, model_str_path="Female_gen_vae_Momenta", sex="female")
copy2clean(model_str='vae', all_num=num_gen, model_str_path="Male_gen_vae_Momenta", sex="male")

### gan
copy2clean(model_str='gan', all_num=num_gen, model_str_path="Female_gen_gan_Momenta", sex="female")
copy2clean(model_str='gan', all_num=num_gen, model_str_path="Male_gen_gan_Momenta", sex="male")


