import pandas as pd
import numpy as np
from utils.compute_mesh_volumes import compute_mass_volume
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
from scipy.stats import wasserstein_distance
from scipy.stats import entropy


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



params = {
            'axes.labelsize': 15.4,
            'font.size': 15.4,
            'legend.fontsize': 15.4,
            'xtick.labelsize': 15.4,
            'ytick.labelsize': 15.4,
            'text.usetex': False,
            'axes.linewidth': 2,
            'xtick.major.width': 2,
            'ytick.major.width': 2,
            'xtick.major.size': 2,
            'ytick.major.size': 2
        }

plt.rcParams.update(params)
font_path = r'C:\Users\kkevopoulos\AppData\Local\Microsoft\Windows\Fonts\SourceSansPro-Regular.otf'
font_prop = font_manager.FontProperties(fname=font_path)
rcParams['font.family'] = font_prop.get_name()

input_dir_general = r"C:\Users\kkevopoulos\Documents\Meshes_Anatomies_Alternative_Branch\Gen_Metrics_Exp_Files_Tagged"

models_dir = ['Targeted_nf', 'Targeted_vae2', 'Targeted_vae3']


x_confounders = pd.read_excel(r"C:\Users\kkevopoulos\OneDrive - Delft University of Technology\Bureaublad\data_kostas_bivme\metadata_final.xlsx")
x_confounders.drop(['Participant ID', 'Height', 'Weight', 'Diastolic BP',
                    'Systolic BP', 'Unnamed: 8', 'Unnamed: 9', 'subject_id'], axis=1, inplace=True)
x_confounders = pd.get_dummies(x_confounders, columns=['Sex'])
x_confounders['Sex_Female'] = x_confounders['Sex_Female'].replace({True: 1, False: 0})
x_confounders['Sex_Male'] = x_confounders['Sex_Male'].replace({True: 1, False: 0})
x_confounders = x_confounders[['BMI', 'Age', 'Sex_Female', 'Sex_Male']]
x_confounders = x_confounders.to_numpy()

### Delete outliers and participants that withdrew from the study
x_confounders = np.delete(x_confounders, [1746, 1831], axis=0)

outliers = np.load('../utils/outliers_indices.npy')
mask = np.ones(x_confounders.shape[0], dtype=bool)
mask[outliers] = False

x_confounders = x_confounders[mask]

### Find the indices of the real subgroup
male_indices = np.where(x_confounders[:, 2] == 0)[0]
x_conf_male = x_confounders[male_indices]

subgroup_indices = np.where(x_conf_male[:, 1] > 58)[0]

### clinical metrics for real anatomies
df_real = pd.read_pickle('../data_models_saved/data/dataframes_clinical_info/df_real_clinical.pkl')
df_real = df_real.iloc[subgroup_indices]


dfs_generated = []
for mod_dir in models_dir:
    df = compute_mass_volume(input_dir=input_dir_general + fr"\{mod_dir}_PCs", num_samples=650, real=True)
    dfs_generated.append(df)

df_nf = dfs_generated[0]
df_vae2 = dfs_generated[1]
df_vae3 = dfs_generated[2]


biomarkers = ['RV_Vol_mL', 'LV_Vol_mL', 'Myo_Mass_g']

biomarkers_wass = []
biomarkers_kl = []
for biom in biomarkers:
    x = np.array(df_real[biom])

    y_nf = np.array(df_nf[biom])
    y_vae2 = np.array(df_vae2[biom])
    y_vae3 = np.array(df_vae3[biom])

    wass_nf = wasserstein_distance(x, y_nf)
    wass_vae2 = wasserstein_distance(x, y_vae2)
    wass_vae3 = wasserstein_distance(x, y_vae3)



    kl_nf = kl_div_histogram(x=x, y=y_nf)
    kl_vae2 = kl_div_histogram(x=x, y=y_vae2)
    kl_vae3 = kl_div_histogram(x=x, y=y_vae3)



    biomarkers_wass.append([wass_nf, wass_vae2, wass_vae3])
    biomarkers_kl.append([kl_nf, kl_vae2, kl_vae3])

    print(f'{biom}: {biomarkers_wass}\n {biomarkers_kl}')

biomarkers_wass = np.array(biomarkers_wass)
biomarkers_kl = np.array(biomarkers_kl)

np.save("../data_models_saved/data/biomarkers_wass_targeted.npy", biomarkers_wass)
np.save("../data_models_saved/data/biomarkers_kl_targeted.npy", biomarkers_kl)

