import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
from scipy.stats import gaussian_kde
import os


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

### Load the sampled metadata for generation
sampled_confounders_female = np.load("../data_models_saved/data/metadata_sampled_female.npy")
sampled_confounders_male = np.load("../data_models_saved/data/metadata_sampled_male.npy")
gen_indices = np.load("../data_models_saved/data/indices_metadata_clinical.npy")

folder = "figures_experiments/t_sne_approximations/"
os.makedirs("../figures_experiments/pheno_dists_contours/", exist_ok=True)

clinical_phenotypes = ['RV_Vol_mL', 'LV_Vol_mL', 'Myo_Mass_g']
models = ['real', 'nf', 'vae1', 'vae2', 'vae3', 'vae4', 'vae5', 'vae6']
for i, model in enumerate(models):
    df = pd.read_pickle(f"../data_models_saved/data/dataframes_clinical_info/df_{model}_clinical.pkl")

    if model == 'real':
        x_confounders_model = x_confounders[df["Index"]]

        female_ind = np.where(x_confounders_model[:, 2] == 1)[0]
        male_ind = np.where(x_confounders_model[:, 2] == 0)[0]
    else:
        female_ind = df["Index"][:gen_indices[2 * (i - 1)]]
        male_ind = df["Index"][gen_indices[2 * (i - 1)]:]

        x_confounders_model_female = sampled_confounders_female[female_ind]
        x_confounders_model_male = sampled_confounders_male[male_ind]

        x_confounders_model = np.concatenate([x_confounders_model_female, x_confounders_model_male], axis=0)

    for pheno in clinical_phenotypes:

        if pheno == 'RV_Vol_mL':
            colormap = 'YlGn'
        elif pheno == 'LV_Vol_mL':
            colormap = 'OrRd'
        elif pheno == 'Myo_Mass_g':
            colormap = 'Blues'

        ### visualize for bmi
        create_plot(metadata_vals=x_confounders_model[:, 0], pheno_vals=df[pheno], metadata_type_str='BMI',
                    model_str=model, pheno_str=pheno, real_metadata_vals_max=max_bmi_real,
                    real_metadata_vals_min=min_bmi_real, real_pheno_vals_max=max(df_real[pheno]),
                    real_pheno_vals_min=min(df_real[pheno]), colormap=colormap)

        ### visualize for age
        create_plot(metadata_vals=x_confounders_model[:, 1], pheno_vals=df[pheno], metadata_type_str='Age',
                    model_str=model, pheno_str=pheno, real_metadata_vals_max=max_age_real,
                    real_metadata_vals_min=min_age_real,real_pheno_vals_max=max(df_real[pheno]),
                    real_pheno_vals_min=min(df_real[pheno]), colormap=colormap)





