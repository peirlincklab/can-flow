import numpy as np
import os
from tqdm import tqdm
import shutil


def write_cp_momenta(base_folder, array_towrite, type):
    file_path = os.path.join(base_folder, f"{type}.txt")

    if type=="Momenta":
        # Open the file and write custom content
        with open(file_path, 'w') as file:
            # Write the first row
            file.write(f"1 720 3\n")
            # Write an empty row
            file.write("\n")

    # Append the numpy array with tab-delimited values
    with open(file_path, 'a') as file:
        np.savetxt(file, array_towrite, delimiter=' ')


def save_momenta(type_momenta, momenta_tosave):
    cp = np.loadtxt("../data_models_saved/data/DeterministicAtlas__EstimatedParameters__ControlPoints.txt")

    # documents_path = '../models_saved/data/momenta2shape'
    documents_path = r'C:\Users\kkevopoulos\Documents\Meshes_Anatomies_Alternative_Branch\Ablation_anatomies'
    base_folder = os.path.join(documents_path, f"{type_momenta}_Momenta")

    pbar = tqdm(total=momenta_tosave.shape[0], desc="Creating folders and files...")
    for i in range(momenta_tosave.shape[0]):
        ### create a folder in Documents called Momenta_i
        subfolder = os.path.join(base_folder, f"Shooting_Momenta_{i}")
        data_folder = os.path.join(subfolder, "data")

        os.makedirs(data_folder, exist_ok=True)

        ###For control points
        write_cp_momenta(base_folder=data_folder, array_towrite=cp, type="ControlPoints")

        ###For momenta (same name of different files, but different momenta are in different files. Just name is the same)
        write_cp_momenta(base_folder=data_folder, array_towrite=momenta_tosave[i], type="Momenta")

        vtk_file = os.path.join(documents_path, "template.vtk")
        destination_file = os.path.join(data_folder, "template.vtk")

        shutil.copy(vtk_file, destination_file)

        model_file = os.path.join(documents_path, 'model.xml')
        destination_file = os.path.join(subfolder, 'model.xml')

        shutil.copy(model_file, destination_file)

        pbar.update()
    pbar.close()
