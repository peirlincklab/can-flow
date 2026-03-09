import numpy as np
from scipy.stats import wasserstein_distance
from scipy.stats import entropy
import pandas as pd
from utils.compute_mesh_volumes import compute_mass_volume
import pickle


def compute_mean_std(num_models, data):
    means = []
    stds = []

    for i in range(0, num_models, 3):
        chunk = data[i:i + 3]
        means.append(np.mean(chunk))
        stds.append(np.std(chunk))

    return means, stds



models_dir_activation = ['cnf_model_ablation_activation_elu_0',
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
                         'cnf_model_ablation_activation_silu_2']

models_dir_n_flow = ['cnf_model_ablation_n_flow_5_0',
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
                     'cnf_model_ablation_n_flow_30_2']

models_dir_out_dim_conf = ['cnf_model_ablation_out_dim_conf_6_0',
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
                           'cnf_model_ablation_out_dim_conf_21_2']


models_dirs = [models_dir_activation, models_dir_n_flow, models_dir_out_dim_conf]
models_dirs_str = ['activation', 'n_flow', 'out_dim_conf']

for variation, variation_str in zip(models_dirs, models_dirs_str):

    cov_list = []
    mmd_list = []

    values_cov = []
    values_mmd = []
    for mod_dir in variation:

        cov = np.load(f'../ablation/mmd_cov_models/cov_vals_{mod_dir}.npy')
        mmd = np.load(f'../ablation/mmd_cov_models/mmd_vals_{mod_dir}.npy')

        values_cov.append(cov)
        values_mmd.append(mmd)

    means_cov, stds_cov = compute_mean_std(num_models=len(variation), data=values_cov)
    means_mmd, stds_mmd = compute_mean_std(num_models=len(variation), data=values_mmd)

    cov_list.append([means_cov, stds_cov])
    mmd_list.append([means_mmd, stds_mmd])

    np.save(f"../ablation/cov_{variation_str}.npy", np.array(cov_list))
    np.save(f"../ablation/mmd_{variation_str}.npy", np.array(mmd_list))



