import pyvista as pv
import numpy as np
import pandas as pd
from tqdm import tqdm


def mesh_extract_region(mesh, region_id_val):
    mask = mesh.cell_data['region_id'] == region_id_val
    cell_ids = np.where(mask)[0]
    subset = mesh.extract_cells(cell_ids)
    # plotter = pv.Plotter()
    # plotter.add_mesh(mesh, color="lightgray", opacity=0.2)
    # plotter.add_mesh(subset, color="red")
    # plotter.show()

    return subset


def compute_mass_volume(input_dir, num_samples, real=False):
    DENSITY = 1.05

    if real:
        mesh_files = [input_dir+fr"\tagged_surface_{i}.vtk" for i in range(num_samples)]
    else:
        mesh_files_female = [input_dir+fr"\tagged_surface_{i}_female.vtk" for i in range(num_samples)]
        mesh_files_male = [input_dir + fr"\tagged_surface_{i}_male.vtk" for i in range(num_samples)]

        mesh_files = mesh_files_female + mesh_files_male

    results = []

    pbar = tqdm(total=len(mesh_files), leave=True, desc='Computing mass/volume...')
    for i, filename in enumerate(mesh_files):

        mesh = pv.read(filename)

        lv_endo = mesh_extract_region(mesh, region_id_val=7)
        surface_lv_endo = lv_endo.extract_surface()
        sealed_lv_endo = surface_lv_endo.fill_holes(np.inf)
        lv_endo_vol = sealed_lv_endo.volume / 1000.0

        rv_endo = mesh_extract_region(mesh, region_id_val=6)
        surface_rv_endo = rv_endo.extract_surface()
        sealed_rv_endo = surface_rv_endo.fill_holes(np.inf)
        rv_endo_vol = sealed_rv_endo.volume / 1000.0

        epi = mesh_extract_region(mesh, region_id_val=5)
        surface_epi = epi.extract_surface()
        sealed_epi = surface_epi.fill_holes(np.inf)
        epi_vol = sealed_epi.volume / 1000.0

        myo_vol = epi_vol - (lv_endo_vol + rv_endo_vol)
        myo_mass = myo_vol * DENSITY

        results.append({
            'File': filename,
            'LV_Vol_mL': lv_endo_vol,
            'RV_Vol_mL': rv_endo_vol,
            'Myo_Mass_g': myo_mass,
            'Index': i
        })

        pbar.update()
    pbar.close()

    df = pd.DataFrame(results)

    return df