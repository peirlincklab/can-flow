from utils.measure_mesh_qualities import MMD, Coverage
import numpy as np
import pyvista as pv
from tqdm import tqdm
import random


random.seed(42)
np.random.seed(42)




def load_pcs_gen(path, num_momenta_samples):

    points_all = []

    pbar = tqdm(total=num_momenta_samples, desc='Loading pcs...')
    for i in range(num_momenta_samples):
        filename_female = path + f"/PointCloud_{i}_female.vtk"
        filename_male = path + f"/PointCloud_{i}_male.vtk"

        points_female = pv.read(filename_female).points
        points_female = np.array(points_female)

        points_male = pv.read(filename_male).points
        points_male = np.array(points_male)

        points_all.append(points_female)
        points_all.append(points_male)

        pbar.update()
    pbar.close()

    return  points_all





def load_pcs_real(num_momenta_samples):

    points_all = []

    pbar = tqdm(total=num_momenta_samples, desc='Loading pcs...')
    for i in range(num_momenta_samples):
        path = "/home/kostas/home/Gen_Metrics_Exp_Files/Real_PCs"
        filename = path + f"/PointCloud_{i}.vtk"

        points = pv.read(filename).points
        points = np.array(points)

        points_all.append(points)

        pbar.update()
    pbar.close()

    return  points_all


path_init = "/home/kostas/home/Gen_Metrics_Exp_Files"

models = ['nf', 'vae1', 'vae2', 'vae3', 'vae4', 'vae5', 'vae6']

real_pcs = load_pcs_real(num_momenta_samples=2274)
N_subsample = 600 ### subsample the real dataset, so that the real and generated sets have equal size

num_runs_stoch = 4
mmd_vals_all = []
cov_vals_all = []
for i in range(num_runs_stoch):
    real_pcs_subsampled = random.sample(real_pcs, N_subsample)

    mmd_vals = []
    cov_vals = []
    for model in models:

        ### gen_pcs now will have 300 female and 300 male samples
        gen_pcs = load_pcs_gen(path=path_init+f"/{model}_PCs", num_momenta_samples=300)

        mmd = MMD(S_g=gen_pcs, S_r=real_pcs_subsampled)
        cov = Coverage(S_g=gen_pcs, S_r=real_pcs_subsampled)

        mmd_vals.append(mmd)
        cov_vals.append(cov)

        print(model, mmd, cov)

    mmd_vals_all.append(mmd_vals)
    cov_vals_all.append(cov_vals)

np.save("data_models_saved/data/mmd_vals.npy", np.array(mmd_vals_all))
np.save("data_models_saved/data/cov_vals.npy", np.array(cov_vals_all))
