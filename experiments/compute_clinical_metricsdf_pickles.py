import pandas as pd
import numpy as np
from utils.compute_mesh_volumes import compute_mass_volume
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams, font_manager
import shutil


def filter_real_data_subgroup(sex_subgroup, age_subgroup, bmi_subgroup):
    """
    Given sex, age, and BMI ranges, find the real anatomies that correspond to this metadata range.

    Parameters
    ----------
    sex_subgroup : float
        1 corresponds to females and 0 corresponds to males
    age_subgroup : float
        Threshold of age, based on which we filter subjects younger than this number
    bmi_subgroup : float
        Threshold of BMI, based on which we filter subjects with more BMI than the threshold

    Returns
    -------
    float
        The number of individuals belonging to the population subgroup defined by the given metadata

    """

    x_confounders = pd.read_excel(
        r"C:\Users\kkevopoulos\OneDrive - Delft University of Technology\Bureaublad\data_kostas_bivme\metadata_final.xlsx")
    x_confounders.drop(['Participant ID', 'Height', 'Weight', 'Diastolic BP',
                        'Systolic BP', 'Unnamed: 8', 'Unnamed: 9', 'subject_id'], axis=1, inplace=True)
    x_confounders = pd.get_dummies(x_confounders, columns=['Sex'])
    x_confounders['Sex_Female'] = x_confounders['Sex_Female'].replace({True: 1, False: 0})
    x_confounders['Sex_Male'] = x_confounders['Sex_Male'].replace({True: 1, False: 0})
    x_confounders = x_confounders[['BMI', 'Age', 'Sex_Female', 'Sex_Male']]
    x_confounders = x_confounders.to_numpy()

    sex_indices = np.where(x_confounders[:, 2] == sex_subgroup)[0]
    x_conf = x_confounders[sex_indices]

    age_indices = np.where(x_conf[:, 1] < age_subgroup)[0]
    x_conf = x_conf[age_indices]

    ### these are the bmi indices
    final_indices = np.where(x_conf[:, 0] > bmi_subgroup)[0]

    os.makedirs(r"C:\Users\kkevopoulos\Documents\Meshes_Anatomies\Targeted_Reference", exist_ok=True)
    j = 0
    for i in range(2208):
        if i in final_indices:
            src = rf"C:\Users\kkevopoulos\Documents\Meshes_Anatomies\Reference_Momenta\Shooting_Momenta_{i}"
            dst = rf"C:\Users\kkevopoulos\Documents\Meshes_Anatomies\Targeted_Reference\Shooting_Momenta_{j}"

            shutil.copytree(src, dst)

            j += 1

    return len(final_indices)


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

input_dir_general = r"C:\Users\kkevopoulos\Documents\Meshes_Anatomies_Alternative_Branch"

models_dir = ['nf', 'vae1', 'vae2', 'vae3', 'vae4', 'vae5', 'vae6']

### Compute clinical metrics for real anatomies
input_dir_real = input_dir_general + "\Gen_Metrics_Exp_Files_Tagged\Real_PCs"
df_real = compute_mass_volume(input_dir=input_dir_real, num_samples=2208, real=True)


### For targeted data --> to filter only a subgroup of the overall population
# sex_subgroup = 0
# bmi_subgroup = 24
# age_subgroup = 56
# num_samples_targeted = filter_real_data_subgroup(sex_subgroup=sex_subgroup,
#                                                  age_subgroup=age_subgroup,
#                                                  bmi_subgroup=bmi_subgroup)
# df_real = compute_mass_volume(input_dir=input_dir_general + "\Targeted_Reference",
#                               num_samples=num_samples_targeted)

dfs_generated = []
indices_metadata = []
for mod_dir in models_dir:
    df = compute_mass_volume(input_dir=input_dir_general + rf"\Gen_Metrics_Exp_Files_Tagged\{mod_dir}_PCs", num_samples=300)
    dfs_generated.append(df)
    indices_metadata.append(df.shape[0])

np.save("../data_models_saved/data/indices_metadata_clinical.npy", np.array(indices_metadata))


df_nf = dfs_generated[0]
df_vae1 = dfs_generated[1]
df_vae2 = dfs_generated[2]
df_vae3 = dfs_generated[3]
df_vae4 = dfs_generated[4]
df_vae5 = dfs_generated[5]
df_vae6 = dfs_generated[6]


os.makedirs("../data_models_saved/data/dataframes_clinical_info", exist_ok=True)

### for targeted subgroup
# df_real.to_pickle("../data_models_saved/data/dataframes_clinical_info/df_real_clinical_targeted.pkl")

df_real.to_pickle("../data_models_saved/data/dataframes_clinical_info/df_real_clinical.pkl")
df_nf.to_pickle("../data_models_saved/data/dataframes_clinical_info/df_nf_clinical.pkl")
df_vae1.to_pickle("../data_models_saved/data/dataframes_clinical_info/df_vae1_clinical.pkl")
df_vae2.to_pickle("../data_models_saved/data/dataframes_clinical_info/df_vae2_clinical.pkl")
df_vae3.to_pickle("../data_models_saved/data/dataframes_clinical_info/df_vae3_clinical.pkl")
df_vae4.to_pickle("../data_models_saved/data/dataframes_clinical_info/df_vae4_clinical.pkl")
df_vae5.to_pickle("../data_models_saved/data/dataframes_clinical_info/df_vae5_clinical.pkl")
df_vae6.to_pickle("../data_models_saved/data/dataframes_clinical_info/df_vae6_clinical.pkl")
