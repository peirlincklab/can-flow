import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.stats import entropy



def kl_div_histogram(x, y):
    px, bins = np.histogram(x, bins=1500, density=False)
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



df_real = pd.read_pickle('../data_models_saved/data/dataframes_clinical_info/df_real_clinical.pkl')

df_nf = pd.read_pickle('../data_models_saved/data/dataframes_clinical_info/df_nf_clinical.pkl')
df_vae1 = pd.read_pickle('../data_models_saved/data/dataframes_clinical_info/df_vae1_clinical.pkl')
df_vae2 = pd.read_pickle('../data_models_saved/data/dataframes_clinical_info/df_vae2_clinical.pkl')
df_vae3 = pd.read_pickle('../data_models_saved/data/dataframes_clinical_info/df_vae3_clinical.pkl')
df_vae4 = pd.read_pickle('../data_models_saved/data/dataframes_clinical_info/df_vae4_clinical.pkl')
df_vae5 = pd.read_pickle('../data_models_saved/data/dataframes_clinical_info/df_vae5_clinical.pkl')
df_vae6 = pd.read_pickle('../data_models_saved/data/dataframes_clinical_info/df_vae6_clinical.pkl')


biomarkers = ['RV_Vol_mL', 'LV_Vol_mL', 'Myo_Mass_g']

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