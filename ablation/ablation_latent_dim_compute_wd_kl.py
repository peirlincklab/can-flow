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



def kl_div_histogram(x, y):
    px, bins = np.histogram(x, bins=25, density=False)
    py, _ = np.histogram(y, bins=bins, density=False)

    px = px.astype(float)
    py = py.astype(float)

    # normalize
    px /= px.sum()
    py /= py.sum()

    # avoid zeros
    eps = 1e-12
    px += eps
    py += eps
    px /= px.sum()
    py /= py.sum()

    kl_xy = entropy(px, py)

    return kl_xy


input_dir_general = r"C:\Users\kkevopoulos\Documents\Meshes_Anatomies"

model = ['cvae2', 'cvae3']

models_dirs = [models_dir_activation, models_dir_n_flow, models_dir_out_dim_conf]
models_dirs_str = ['activation', 'n_flow', 'out_dim_conf']


for variation, variation_str in zip(models_dirs, models_dirs_str):

    dfs_generated = []
    for mod_dir in variation:
        df = compute_mass_volume(input_dir=input_dir_general + fr"\{mod_dir}_Momenta", num_samples=600)
        dfs_generated.append(df)


    df_real = pd.read_pickle('../data_models_saved/data/dataframes_clinical_info/df_real_clinical.pkl')

    biomarkers = ['RV_Vol_mL', 'LV_Vol_mL', 'Myo_Mass_g']

    wass_dict = {}
    kl_dict = {}
    for biom in biomarkers:
        x = np.array(df_real[biom])

        values_wass = []
        values_kl = []
        for df_gen in dfs_generated:
            y_df = np.array(df_gen[biom])

            wass = wasserstein_distance(x, y_df)
            kl = kl_div_histogram(x=x, y=y_df)


            values_wass.append(wass)
            values_kl.append(kl)

        means_wass, stds_wass = compute_mean_std(num_models=len(variation), data=values_wass)
        means_kl, stds_kl = compute_mean_std(num_models=len(variation), data=values_kl)

        wass_dict[biom] = [means_wass, stds_wass]
        kl_dict[biom] = [means_kl, stds_kl]

    with open(f"../ablation/wass_dict_{variation_str}.pkl", "wb") as f:
        pickle.dump(wass_dict, f)

    with open(f"../ablation/kl_dict_{variation_str}.pkl", "wb") as f:
        pickle.dump(kl_dict, f)






