import numpy as np
from scipy.stats import entropy
import pandas as pd
from utils.compute_mesh_volumes import compute_mass_volume
import pickle


def kl_div_histogram(x, y, bins=25, eps=1e-12):
    """
        Compute the Kullback-Leibler (KL) divergence between two one-dimensional
        datasets using histogram-based probability estimates.

        First construct a common histogram range based on the minimum
        and maximum values across both input arrays. Then, estimate the empirical
        probability distributions of `x` and `y` using the same bin edges, add a
        small numerical constant to avoid division by zero, and compute the KL
        divergence D_KL(P_x || P_y).

        Parameters
        ----------
        x : array-like
            First input dataset. This distribution is treated as the reference
            distribution P_x in the KL divergence.
        y : array-like
            Second input dataset. This distribution is treated as the comparison
            distribution P_y in the KL divergence.
        bins : int, optional
            Number of histogram bins used to estimate the probability distributions.
            Default is 25.
        eps : float, optional
            Small constant added to the histogram probabilities to avoid numerical
            issues caused by zero probabilities. Default is 1e-12.

        Returns
        -------
        float
            Histogram-based estimate of the KL divergence D_KL(P_x || P_y)
        """
    x = np.asarray(x)
    y = np.asarray(y)

    ### common bin range
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())

    px, bin_edges = np.histogram(x, bins=bins, range=(min_val, max_val), density=False)
    py, _ = np.histogram(y, bins=bin_edges, density=False)

    px = px.astype(float)
    py = py.astype(float)

    px /= px.sum()
    py /= py.sum()

    ### avoid division by zero
    px += eps
    py += eps

    px /= px.sum()
    py /= py.sum()

    return entropy(px, py)



input_dir_general = r"C:\Users\kkevopoulos\Documents\Meshes_Anatomies_Alternative_Branch\Ablation_anatomies\Gen_Metrics_Exp_Files_Ablation"

models = ['canflow', 'cvae2', 'cvae3']
dims = ['10', '20', '30', '40', '50']

for dim in dims:
    dfs_generated = []
    for model in models:
        specific_dir = f'AblationLatentDim_{model}_latent{dim}'

        df = compute_mass_volume(input_dir=input_dir_general + fr"\{specific_dir}_PCs", num_samples=600, ablation=True)
        dfs_generated.append(df)

    df_real = pd.read_pickle('../data_models_saved/data/dataframes_clinical_info/df_real_clinical.pkl')

    biomarkers = ['LV_Vol_mL', 'RV_Vol_mL', 'Myo_Mass_g', 'RVEDV_LVEDV_ratio', 'Long_axis_length', 'LV_Sphericity']

    kl_dict = {}
    for biom in biomarkers:
        x = np.array(df_real[biom])

        values_kl = []
        for df_gen in dfs_generated:
            y_df = np.array(df_gen[biom])

            kl = kl_div_histogram(x=x, y=y_df)

            values_kl.append(kl)

        kl_dict[biom] = values_kl

    with open(f"../ablation/kl_dict_latentdim_{dim}.pkl", "wb") as f:
        pickle.dump(kl_dict, f)









