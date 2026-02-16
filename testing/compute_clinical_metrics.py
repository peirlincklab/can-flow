import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from utils.compute_mesh_volumes import compute_mass_volume
import pandas as pd

def create_plot(metadata_vals, pheno_vals, metadata_type_str,
                model_str, pheno_str, real_metadata_vals_max,
                real_metadata_vals_min, real_pheno_vals_max,
                real_pheno_vals_min, colormap):

    pheno_vals = np.asarray(pheno_vals, dtype=float)
    metadata_vals = np.asarray(metadata_vals, dtype=float)
    # Create grid
    x_min, x_max = (real_metadata_vals_min - 1, real_metadata_vals_max + 1)
    y_min, y_max = (real_pheno_vals_min - 1, real_pheno_vals_max + 1)

    x_grid, y_grid = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400)
    )


    plt.figure(figsize=(8, 7))


    f_kde = gaussian_kde(np.vstack([metadata_vals, pheno_vals]))
    f_plot = f_kde(np.vstack([x_grid.ravel(), y_grid.ravel()])).reshape(x_grid.shape)
    plt.contourf(x_grid, y_grid, f_plot, cmap=colormap, levels=9)

    plt.xlabel(metadata_type_str)
    plt.savefig(f"../figures_experiments/pheno_dists_contours/{model_str}_{pheno_str}_{metadata_type_str}.pdf")
    plt.close()


input_dir_general = r"C:\Users\kkevopoulos\Documents\Meshes_Anatomies"

models_dir = ['Targeted_nf', 'Targeted_vae2', 'Targeted_vae3']


dfs_generated = []
indices_metadata = []
for mod_dir in models_dir:
    df = compute_mass_volume(input_dir=input_dir_general + fr"\{mod_dir}_Momenta", num_samples=750)
    dfs_generated.append(df)
    indices_metadata.append(df.shape[0])

df_nf = dfs_generated[0]
df_vae2 =dfs_generated[1]
df_vae3 = dfs_generated[2]


### firstly visualize for the real data
x_confounders = pd.read_excel(r"C:\Users\kkevopoulos\OneDrive - Delft University of Technology\Bureaublad\data_kostas_bivme\metadata_final.xlsx")
x_confounders.drop(['Participant ID', 'Height', 'Weight', 'Diastolic BP',
                    'Systolic BP', 'Unnamed: 8', 'Unnamed: 9', 'subject_id'], axis=1, inplace=True)
x_confounders = pd.get_dummies(x_confounders, columns=['Sex'])
x_confounders['Sex_Female'] = x_confounders['Sex_Female'].replace({True: 1, False: 0})
x_confounders['Sex_Male'] = x_confounders['Sex_Male'].replace({True: 1, False: 0})
x_confounders = x_confounders[['BMI', 'Age', 'Sex_Female', 'Sex_Male']]
x_confounders = x_confounders.to_numpy()

df_real = pd.read_pickle('../data_models_saved/data/dataframes_clinical_info/df_real_clinical.pkl')
df_real = df_real.drop(index=1131)


min_bmi_real = min(x_confounders[df_real['Index']][:, 0])
max_bmi_real = max(x_confounders[df_real['Index']][:, 0])

min_age_real = min(x_confounders[df_real['Index']][:, 1])
max_age_real = max(x_confounders[df_real['Index']][:, 1])


clinical_phenotypes = ['RV_Vol_mL', 'LV_Vol_mL', 'Myo_Mass_g']
models = ['nf', 'vae2', 'vae3']


