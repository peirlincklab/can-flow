import os
import  numpy as np
import shutil
from tqdm import tqdm


def write_cp_momenta(base_folder, array_towrite, type):
    file_path = os.path.join(base_folder, f"{type}.txt")

    if type == "Momenta":
        # Open the file and write custom content
        with open(file_path, 'w') as file:
            # Write the first row
            file.write(f"1 2730 3\n")
            # Write an empty row
            file.write("\n")

    # Append the numpy array with tab-delimited values
    with open(file_path, 'a') as file:
        np.savetxt(file, array_towrite, delimiter=' ')


def load_cp_momenta(file_name):

    # Get the path to the Downloads folder dynamically
    downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")

    file_path = os.path.join(downloads_folder, file_name)
    data = np.loadtxt(file_path)

    return data



file_name_cp = "cardiac_atlas_kostas\output\DeterministicAtlas__EstimatedParameters__ControlPoints.txt"
file_name_momenta = "cardiac_atlas_kostas\output\DeterministicAtlas__EstimatedParameters__Momenta.txt"


control_points = load_cp_momenta(file_name_cp)
momenta = load_cp_momenta(file_name_momenta)

momenta = np.delete(momenta, 0, axis=0)
momenta = momenta.reshape((456, 2730, 3))


documents_path = os.path.join(os.environ['USERPROFILE'], 'Documents')
base_folder = os.path.join(documents_path, "Reference_Momenta")


pbar = tqdm(total=momenta.shape[0], desc="Creating folders and files...")
for i in range(momenta.shape[0]):
    ### create a folder in Documents called Momenta_i
    subfolder = os.path.join(base_folder, f"Shooting_Momenta_{i}")
    data_folder = os.path.join(subfolder, "data")

    os.makedirs(data_folder, exist_ok=True)

    ###For control points
    write_cp_momenta(base_folder=data_folder, array_towrite=control_points, type="ControlPoints")

    ###For momenta (same name of different files, but different momenta are in different files. Just name is the same)
    write_cp_momenta(base_folder=data_folder, array_towrite=momenta[i], type="Momenta")

    vtk_file = os.path.join(documents_path, "template.vtk")
    destination_file = os.path.join(data_folder, "template.vtk")

    shutil.copy(vtk_file, destination_file)

    model_file = os.path.join(documents_path, 'model.xml')
    destination_file = os.path.join(subfolder, 'model.xml')

    shutil.copy(model_file, destination_file)

    pbar.update()
pbar.close()


