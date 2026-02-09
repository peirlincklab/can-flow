import pandas as pd
from utils.compute_mesh_volumes import compute_mass_volume
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams, font_manager


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

input_dir_general = r"..\data_models_saved\data\momenta2shape"

models_dir = ['Female_gen_nf', 'Male_gen_nf',
              'Female_gen_vae1', 'Male_gen_vae1',
              'Female_gen_vae2', 'Male_gen_vae2',
              'Female_gen_vae3', 'Male_gen_vae3',
              'Female_gen_vae4', 'Male_gen_vae4',
              'Female_gen_vae5', 'Male_gen_vae5',
              'Female_gen_vae6', 'Male_gen_vae6']

### Compute clinical metrics for real anatomies
input_dir_real = input_dir_general + "\Reference_Momenta"

df_real = compute_mass_volume(input_dir=input_dir_real, num_samples=2274)


dfs_generated = []
for mod_dir in models_dir:
    df = compute_mass_volume(input_dir=input_dir_general + fr"\{mod_dir}_Momenta", num_samples=700)
    dfs_generated.append(df)


df_nf = pd.concat([dfs_generated[0], dfs_generated[1]], axis=0, ignore_index=True)
df_vae1 = pd.concat([dfs_generated[2], dfs_generated[3]], axis=0, ignore_index=True)
df_vae2 = pd.concat([dfs_generated[4], dfs_generated[5]], axis=0, ignore_index=True)
df_vae3 = pd.concat([dfs_generated[6], dfs_generated[7]], axis=0, ignore_index=True)
df_vae4 = pd.concat([dfs_generated[8], dfs_generated[9]], axis=0, ignore_index=True)
df_vae5 = pd.concat([dfs_generated[10], dfs_generated[11]], axis=0, ignore_index=True)
df_vae6 = pd.concat([dfs_generated[12], dfs_generated[13]], axis=0, ignore_index=True)



os.makedirs("../data_models_saved/data/dataframes_clinical_info", exist_ok=True)

df_real.to_pickle("../data_models_saved/data/dataframes_clinical_info/df_real_clinical.pkl")
df_nf.to_pickle("../data_models_saved/data/dataframes_clinical_info/df_nf_clinical.pkl")
df_vae1.to_pickle("../data_models_saved/data/dataframes_clinical_info/df_vae1_clinical.pkl")
df_vae2.to_pickle("../data_models_saved/data/dataframes_clinical_info/df_vae2_clinical.pkl")
df_vae3.to_pickle("../data_models_saved/data/dataframes_clinical_info/df_vae3_clinical.pkl")
df_vae4.to_pickle("../data_models_saved/data/dataframes_clinical_info/df_vae4_clinical.pkl")
df_vae5.to_pickle("../data_models_saved/data/dataframes_clinical_info/df_vae5_clinical.pkl")
df_vae6.to_pickle("../data_models_saved/data/dataframes_clinical_info/df_vae6_clinical.pkl")
