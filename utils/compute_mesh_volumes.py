import pyvista as pv
import numpy as np
import os
import pandas as pd
from tqdm import tqdm


def debug_plot(mesh, title, points=None):
    p = pv.Plotter()
    p.add_text(title)
    p.add_mesh(mesh, show_edges=True)

    if points is not None:
        p.add_points(
            points,
            color="red",
            point_size=15,
            render_points_as_spheres=True
        )

    p.show()


### New meshes - clipping with pyvista and sealing the cavities
# input_dir = r"C:\Users\kkevopoulos\Documents\Meshes_Anatomies\Reference_Momenta"

def compute_mass_volume(input_dir, num_samples):
    DENSITY = 1.05
    Z_OFFSET = 20

    percentage = 0.285

    mesh_files = [input_dir+f"\Shooting_Momenta_{i}\output\Shooting__GeodesicFlow__biv__tp_10__age_1.00.vtk" for i in range(num_samples)]


    results = []

    pbar = tqdm(total=len(mesh_files), leave=True, desc='Computing mass/volume...')
    for i, filename in enumerate(mesh_files):
        if filename.endswith('.vtk'):
            mesh = pv.read(filename)

            # Orient with the z-axis (I measured this with Paraview)
            transformed = mesh.copy()
            transformed.translate([-19.408, -7.14409, 31.259], inplace=True)
            transformed.rotate_x(-29.1004, inplace=True)
            transformed.rotate_y(-56.7632, inplace=True)
            transformed.rotate_z(-59.9904, inplace=True)

            z_min, z_max = transformed.bounds[4], transformed.bounds[5]
            z_length = z_max - z_min

            clip_z = z_max - percentage * z_length

            # --- 2. Dynamic Clipping with Perfect Sealing ---
            com = transformed.center_of_mass()
            # clip_origin = (com[0], com[1], com[2] + Z_OFFSET)
            clip_origin = (0.0, 0.0, clip_z)

            clipped = transformed.clip(normal='z', origin=clip_origin, invert=True)

            # Splitting the clipped mesh in epicardium, LV, and RV
            bodies = clipped.connectivity().split_bodies()

            processed_data = []
            for b in bodies:
                surface_body = b.extract_surface()
                sealed = surface_body.fill_holes(np.inf)

                sealed = sealed.compute_normals(
                    auto_orient_normals=True,
                    consistent_normals=True,
                    inplace=False
                )


                processed_data.append({
                    'volume': sealed.volume / 1000.0,
                    'centroid': sealed.center,
                    'mesh_obj': sealed
                })

            # Sort based on the volume
            processed_data.sort(key=lambda x: x['volume'], reverse=True)
            if len(processed_data) == 3:
                v_epi_data = processed_data[0]

                # Sort the two cavities by X-axis (Centroid)
                cavities = processed_data[1:3]
                cavities.sort(key=lambda x: x['centroid'][0])

                # Identify the chambers based on the orientation
                lv_data = cavities[0]
                rv_data = cavities[1]

                myo_vol = v_epi_data['volume'] - (lv_data['volume'] + rv_data['volume'])
                myo_mass = myo_vol * DENSITY

                # Plot to check if the surfaces have been capped correctly

                # p = pv.Plotter(shape=(1, 4), title=f"Analysis: {filename}")
                #
                # plane_zmin = pv.Plane(
                #     center=(0, 0, z_min),
                #     direction=(0, 0, 1),
                #     i_size=200,
                #     j_size=200
                # )
                #
                # plane_zmax = pv.Plane(
                #     center=(0, 0, z_max),
                #     direction=(0, 0, 1),
                #     i_size=200,
                #     j_size=200
                # )
                #
                # plane_clip = pv.Plane(
                #     center=(0, 0, clip_z),
                #     direction=(0, 0, 1),
                #     i_size=200,
                #     j_size=200
                # )
                #
                # p.subplot(0, 0)
                # p.add_text("1. Clipped Mesh", font_size=10)
                # p.add_mesh(plane_clip, color="yellow", opacity=0.6, label="clip plane")
                # p.add_mesh(clipped, color="lightgrey", opacity=0.5, show_edges=True)
                #
                # p.add_mesh(plane_zmin, color="green", opacity=0.4, label="z_min")
                # p.add_mesh(plane_zmax, color="red", opacity=0.4, label="z_max")
                #
                #
                #
                # # Panel 1: Total Clipped Model (Epicardium)
                # p.subplot(0, 1)
                # p.add_text("1. Clipped Mesh", font_size=10)
                # p.add_mesh(clipped, color="lightgrey", opacity=0.5, show_edges=True)
                #
                # # Panel 2: The Cavities (LV and RV)
                # p.subplot(0, 2)
                # p.add_text(f"2. Cavities\nLV: {lv_data['volume']:.1f}mL | RV: {rv_data['volume']:.1f}mL", font_size=10)
                # p.add_mesh(lv_data['mesh_obj'], color="red", label="LV")
                # p.add_mesh(rv_data['mesh_obj'], color="blue", label="RV")
                #
                # # Panel 3: Myocardium Visualization
                # # We show the Myocardium by plotting the Epicardium and "cutting out" the cavities
                # p.subplot(0, 3)
                # p.add_text(f"3. Myocardium\nMass: {myo_mass:.1f}g", font_size=10)
                # p.add_mesh(v_epi_data['mesh_obj'], color="pink", opacity=0.3)  # The outer boundary
                # p.add_mesh(lv_data['mesh_obj'], color="white", opacity=1.0)  # Visual "hole" for LV
                # p.add_mesh(rv_data['mesh_obj'], color="white", opacity=1.0)  # Visual "hole" for RV
                #
                # p.link_views()
                # p.show()



                results.append({
                    'File': filename,
                    'LV_Vol_mL': lv_data['volume'],
                    'RV_Vol_mL': rv_data['volume'],
                    'Myo_Mass_g': myo_mass,
                    'Index': i
                })
                # print(f"Processed {filename}")
            else:
                print(f"Error: Found only {len(processed_data)} bodies in {filename}")
        pbar.update()
    pbar.close()
    df = pd.DataFrame(results)

    return df