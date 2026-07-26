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
        filename = path + f"/PointCloud_{i}.vtk"

        points = pv.read(filename).points
        points = np.array(points)

        points_all.append(points)

        pbar.update()
    pbar.close()

    return  points_all





def load_pcs_real(num_momenta_samples):

    points_all = []

    pbar = tqdm(total=num_momenta_samples, desc='Loading pcs...')
    for i in range(num_momenta_samples):
        path = "/home/kostas/home/Gen_Metrics_Exp_Files/Targeted_Real_PCs"
        filename = path + f"/PointCloud_{i}.vtk"

        points = pv.read(filename).points
        points = np.array(points)

        points_all.append(points)

        pbar.update()
    pbar.close()

    return  points_all


path_init = "/home/kostas/home/Gen_Metrics_Exp_Files"

models = ['nf', 'vae2', 'vae3']

real_pcs = load_pcs_real(num_momenta_samples=699)
N_subsample = 650 ### subsample the real dataset, so that the real and generated sets have equal size

num_runs_stoch = 4
mmd_vals_all = []
cov_vals_all = []
for i in range(num_runs_stoch):
    real_pcs_subsampled = random.sample(real_pcs, N_subsample)

    mmd_vals = []
    cov_vals = []
    for model in models:

        ### gen_pcs now will have 650 samples
        gen_pcs = load_pcs_gen(path=path_init+f"/Targeted_{model}_PCs", num_momenta_samples=650)

        mmd = MMD(S_g=gen_pcs, S_r=real_pcs_subsampled)
        cov = Coverage(S_g=gen_pcs, S_r=real_pcs_subsampled)

        mmd_vals.append(mmd)
        cov_vals.append(cov)

        print(model, mmd, cov)

    mmd_vals_all.append(mmd_vals)
    cov_vals_all.append(cov_vals)

np.save("models_saved/data/mmd_vals_targeted.npy", np.array(mmd_vals_all))
np.save("models_saved/data/cov_vals_targeted.npy", np.array(cov_vals_all))
