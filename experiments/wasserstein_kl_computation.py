import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.stats import entropy

### This file computes WD and KL divergence between real and synthetic distributions of clinical phenotypes

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


### load all the dfs that contain information about the clinical phenotypes
df_real = pd.read_pickle('../data_models_saved/data/dataframes_clinical_info/df_real_clinical.pkl')

df_nf = pd.read_pickle('../data_models_saved/data/dataframes_clinical_info/df_nf_clinical.pkl')
df_vae1 = pd.read_pickle('../data_models_saved/data/dataframes_clinical_info/df_vae1_clinical.pkl')
df_vae2 = pd.read_pickle('../data_models_saved/data/dataframes_clinical_info/df_vae2_clinical.pkl')
df_vae3 = pd.read_pickle('../data_models_saved/data/dataframes_clinical_info/df_vae3_clinical.pkl')
df_vae4 = pd.read_pickle('../data_models_saved/data/dataframes_clinical_info/df_vae4_clinical.pkl')
df_vae5 = pd.read_pickle('../data_models_saved/data/dataframes_clinical_info/df_vae5_clinical.pkl')
df_vae6 = pd.read_pickle('../data_models_saved/data/dataframes_clinical_info/df_vae6_clinical.pkl')


biomarkers = ['RV_Vol_mL', 'LV_Vol_mL', 'Myo_Mass_g', 'RVEDV_LVEDV_ratio', 'Long_axis_length', 'LV_Sphericity']

biomarkers_wass = []
biomarkers_kl = []
for biom in biomarkers:
    x = np.array(df_real[biom])

    y_nf = np.array(df_nf[biom])
    y_vae1 = np.array(df_vae1[biom])
    y_vae2 = np.array(df_vae2[biom])
    y_vae3 = np.array(df_vae3[biom])
    y_vae4 = np.array(df_vae4[biom])
    y_vae5 = np.array(df_vae5[biom])
    y_vae6 = np.array(df_vae6[biom])

    wass_nf = wasserstein_distance(x, y_nf)
    wass_vae1 = wasserstein_distance(x, y_vae1)
    wass_vae2 = wasserstein_distance(x, y_vae2)
    wass_vae3 = wasserstein_distance(x, y_vae3)
    wass_vae4 = wasserstein_distance(x, y_vae4)
    wass_vae5 = wasserstein_distance(x, y_vae5)
    wass_vae6 = wasserstein_distance(x, y_vae6)


    kl_nf = kl_div_histogram(x=x, y=y_nf)
    kl_vae1 = kl_div_histogram(x=x, y=y_vae1)
    kl_vae2 = kl_div_histogram(x=x, y=y_vae2)
    kl_vae3 = kl_div_histogram(x=x, y=y_vae3)
    kl_vae4 = kl_div_histogram(x=x, y=y_vae4)
    kl_vae5 = kl_div_histogram(x=x, y=y_vae5)
    kl_vae6 = kl_div_histogram(x=x, y=y_vae6)


    biomarkers_wass.append([wass_nf, wass_vae1, wass_vae2, wass_vae3, wass_vae4, wass_vae5, wass_vae6])
    biomarkers_kl.append([kl_nf, kl_vae1, kl_vae2, kl_vae3, kl_vae4, kl_vae5, kl_vae6])

    print(f'{biom}: {biomarkers_wass}\n {biomarkers_kl}')

biomarkers_wass = np.array(biomarkers_wass)
biomarkers_kl = np.array(biomarkers_kl)

np.save("../data_models_saved/data/biomarkers_wass.npy", biomarkers_wass)
np.save("../data_models_saved/data/biomarkers_kl.npy", biomarkers_kl)