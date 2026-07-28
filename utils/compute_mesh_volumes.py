import pyvista as pv
import numpy as np
import pandas as pd
from tqdm import tqdm



def fit_plane_normal(points):
    """
    Fit a plane to points and return its normal vector.
    The normal is the last principal component.
    """
    X = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    normal = vh[-1]
    return normal / np.linalg.norm(normal)


def compute_lv_long_axis_from_open_base(lv_mesh):
    """
    Compute LV long-axis length from an isolated LV surface mesh.

    Assumption:
    - The LV mesh is open at the mitral valve/base.
    - The largest boundary loop corresponds to the mitral valve.
    """

    # Convert to clean triangular surface
    lv = lv_mesh.extract_surface().triangulate().clean()

    # Extract open boundary edges
    boundary = lv.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )

    # Split boundary into connected loops
    boundary = boundary.connectivity()

    if "region_id" not in boundary.cell_data:
        raise RuntimeError("Could not find connected boundary regions.")

    region_ids = boundary.cell_data["region_id"]
    unique_ids, counts = np.unique(region_ids, return_counts=True)

    # Usually, the largest open boundary is the mitral annulus
    mitral_region = unique_ids[np.argmax(counts)]
    annulus = boundary.extract_cells(region_ids == mitral_region)

    zero_cell_ids = np.where(annulus.cell_data['RegionId'] == 0)[0]
    one_cell_ids = np.where(annulus.cell_data['RegionId'] == 1)[0]

    if zero_cell_ids.shape[0] > one_cell_ids.shape[0]:
        annulus_chosen = annulus.extract_cells(zero_cell_ids)
    else:
        annulus_chosen = annulus.extract_cells(one_cell_ids)

    annulus_points = annulus_chosen.points

    # Mitral annulus center
    base_center = annulus_points.mean(axis=0)

    # Fit mitral annulus plane
    base_normal = fit_plane_normal(annulus_points)

    # Signed distance of all LV points from the mitral plane
    signed_distances = (lv.points - base_center) @ base_normal

    # Apex is the point farthest from the mitral annulus plane
    apex_id = np.argmax(np.abs(signed_distances))
    apex = lv.points[apex_id]

    # Long-axis length: apex to mitral annulus center
    long_axis_length = np.linalg.norm(apex - base_center)

    return long_axis_length, apex, base_center, annulus_chosen, lv




def mesh_extract_region(mesh, region_id_val):
    mask = mesh.cell_data['region_id'] == region_id_val
    cell_ids = np.where(mask)[0]
    subset = mesh.extract_cells(cell_ids)
    # plotter = pv.Plotter()
    # plotter.add_mesh(mesh, color="lightgray", opacity=0.2)
    # plotter.add_mesh(subset, color="red")
    # plotter.show()

    return subset


def compute_mass_volume(input_dir, num_samples, real=False, ablation=False):
    DENSITY = 1.05

    if real:
        mesh_files = [input_dir+fr"\tagged_surface_{i}.vtk" for i in range(num_samples)]
    else:
        if not ablation:
            mesh_files_female = [input_dir+fr"\tagged_surface_{i}_female.vtk" for i in range(num_samples)]
            mesh_files_male = [input_dir + fr"\tagged_surface_{i}_male.vtk" for i in range(num_samples)]

            mesh_files = mesh_files_female + mesh_files_male
        else:
            mesh_files = [input_dir + fr"\PointCloud_{i}.vtk" for i in range(num_samples)]

    results = []

    pbar = tqdm(total=len(mesh_files), leave=True, desc='Computing mass/volume...')
    for i, filename in enumerate(mesh_files):

        mesh = pv.read(filename)

        lv_endo = mesh_extract_region(mesh, region_id_val=7)

        L_long_axis_length, apex, base_center, annulus, lv = compute_lv_long_axis_from_open_base(lv_endo)

        # print("LV long-axis length:", L)
        # print("Apex:", apex)
        # print("Base center:", base_center)
        #
        # line = pv.Line(apex, base_center)
        #
        # p = pv.Plotter()
        # p.add_mesh(lv, opacity=0.35)
        # p.add_mesh(annulus, line_width=5)
        # p.add_mesh(line, line_width=6)
        # p.add_points(np.array([apex]), point_size=15, render_points_as_spheres=True)
        # p.add_points(np.array([base_center]), point_size=15, render_points_as_spheres=True)
        # p.show()

        surface_lv_endo = lv_endo.extract_surface()
        sealed_lv_endo = surface_lv_endo.fill_holes(np.inf)
        lv_endo_vol = sealed_lv_endo.volume / 1000.0

        lv_sphericity = lv_endo_vol /  ( (np.pi/6) * L_long_axis_length**3 )

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
            'RVEDV_LVEDV_ratio': (rv_endo_vol/lv_endo_vol) * 100,
            'Long_axis_length': L_long_axis_length,
            'LV_Sphericity': lv_sphericity * 100000,
            'Index': i
        })

        pbar.update()
    pbar.close()

    df = pd.DataFrame(results)

    return df