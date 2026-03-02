from utils.measure_mesh_qualities import MMD, Coverage
import numpy as np
import pyvista as pv
from tqdm import tqdm
import random
import os

os.makedirs('ablation/mmd_cov_models', exist_ok=True)


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
        path = "/home/kostas/home/Gen_Metrics_Exp_Files/Real_PCs"
        filename = path + f"/PointCloud_{i}.vtk"

        points = pv.read(filename).points
        points = np.array(points)

        points_all.append(points)

        pbar.update()
    pbar.close()

    return  points_all


path_init = "/home/kostas/home/Gen_Metrics_Exp_Files_Ablation" ### this folder should exist in

models = ['cnf_model_ablation_activation_elu_0',
             'cnf_model_ablation_activation_elu_1',
             'cnf_model_ablation_activation_elu_2',
             'cnf_model_ablation_activation_leaky_relu_0',
             'cnf_model_ablation_activation_leaky_relu_1',
             'cnf_model_ablation_activation_leaky_relu_2',
             'cnf_model_ablation_activation_relu_0',
             'cnf_model_ablation_activation_relu_1',
             'cnf_model_ablation_activation_relu_2',
             'cnf_model_ablation_activation_silu_0',
             'cnf_model_ablation_activation_silu_1',
             'cnf_model_ablation_activation_silu_2',
              'cnf_model_ablation_n_flow_5_0',
              'cnf_model_ablation_n_flow_5_1',
              'cnf_model_ablation_n_flow_5_2',
              'cnf_model_ablation_n_flow_10_0',
              'cnf_model_ablation_n_flow_10_1',
              'cnf_model_ablation_n_flow_10_2',
              'cnf_model_ablation_n_flow_20_0',
              'cnf_model_ablation_n_flow_20_1',
              'cnf_model_ablation_n_flow_20_2',
              'cnf_model_ablation_n_flow_25_0',
              'cnf_model_ablation_n_flow_25_1',
              'cnf_model_ablation_n_flow_25_2',
              'cnf_model_ablation_n_flow_30_0',
              'cnf_model_ablation_n_flow_30_1',
              'cnf_model_ablation_n_flow_30_2',
               'cnf_model_ablation_out_dim_conf_6_0',
               'cnf_model_ablation_out_dim_conf_6_1',
               'cnf_model_ablation_out_dim_conf_6_2',
               'cnf_model_ablation_out_dim_conf_9_0',
               'cnf_model_ablation_out_dim_conf_9_1',
               'cnf_model_ablation_out_dim_conf_9_2',
               'cnf_model_ablation_out_dim_conf_15_0',
               'cnf_model_ablation_out_dim_conf_15_1',
               'cnf_model_ablation_out_dim_conf_15_2',
               'cnf_model_ablation_out_dim_conf_18_0',
               'cnf_model_ablation_out_dim_conf_18_1',
               'cnf_model_ablation_out_dim_conf_18_2',
               'cnf_model_ablation_out_dim_conf_21_0',
               'cnf_model_ablation_out_dim_conf_21_1',
               'cnf_model_ablation_out_dim_conf_21_2'
          ]

real_pcs = load_pcs_real(num_momenta_samples=2274)
N_subsample = 600 ### subsample the real dataset, so that the real and generated sets have equal size
num_runs_stoch = 3

for num_model, model in enumerate(models):
    mmd_vals = []
    cov_vals = []

    for i in range(num_runs_stoch):
        real_pcs_subsampled = random.sample(real_pcs, N_subsample)

        ### gen_pcs now will have 600 samples
        gen_pcs = load_pcs_gen(path=path_init+f"/{model}_PCs", num_momenta_samples=600)

        mmd = MMD(S_g=gen_pcs, S_r=real_pcs_subsampled)
        cov = Coverage(S_g=gen_pcs, S_r=real_pcs_subsampled)

        mmd_vals.append(mmd)
        cov_vals.append(cov)


    print(f"DONE: Model {num_model}/{len(models)}")
    np.save(f"ablation/mmd_cov_models/mmd_vals_{model}.npy", np.mean(mmd_vals))
    np.save(f"ablation/mmd_cov_models/cov_vals_{model}.npy", np.mean(cov_vals))
