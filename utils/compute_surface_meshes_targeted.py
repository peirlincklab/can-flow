import pyvista as pv
import os
from tqdm import tqdm

os.makedirs(r"C:\Users\kkevopoulos\Documents\Meshes_Anatomies_Alternative_Branch\Gen_Metrics_Exp_Files_Tagged", exist_ok=True)

general_path = r"C:\Users\kkevopoulos\Documents\Meshes_Anatomies_Alternative_Branch"
template_tagged = pv.read(r"C:\Users\kkevopoulos\Documents\Meshes_Anatomies_Alternative_Branch\template_tagged_correct.vtk")

path_tosave = r"C:\Users\kkevopoulos\Documents\Meshes_Anatomies_Alternative_Branch\Gen_Metrics_Exp_Files_Tagged"


models = ['nf', 'vae2', 'vae3']

for model in models:
    num_gen = 650
    general_path_generated = general_path + rf"\Gen_Metrics_Exp_Files\Targeted_{model}_PCs"

    os.makedirs(path_tosave + fr'\Targeted_{model}_PCs', exist_ok=True)

    pbar_gen = tqdm(total=num_gen, desc='Tagging generated anatomies...', leave=True)
    for i in range(num_gen):

        mesh = pv.read(general_path_generated + f'\PointCloud_{i}.vtk')

        mesh.cell_data['region_id'] = template_tagged.cell_data['region_id']

        # save
        mesh.save(path_tosave + rf'\Targeted_{model}_PCs\tagged_surface_{i}.vtk')

        pbar_gen.update()
    pbar_gen.close()

### Do the same for the real anatomies
num_real = 2208
general_path_real = general_path + f"\Gen_Metrics_Exp_Files\Targeted_Real_PCs"
os.makedirs(path_tosave + '\Targeted_Real_PCs', exist_ok=True)
pbar_real = tqdm(total=num_real, desc='Tagging real anatomies...', leave=True)
for i in range(num_real):
    real_mesh = pv.read(general_path_real + f'\PointCloud_{i}.vtk')

    real_mesh.cell_data['region_id'] = template_tagged.cell_data['region_id']

    real_mesh.save(path_tosave + rf'\Targeted_Real_PCs\tagged_surface_{i}.vtk')

    pbar_real.update()
pbar_real.close()











