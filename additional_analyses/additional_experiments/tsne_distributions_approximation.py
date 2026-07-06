### Experiment -- compare the real and synthetic distributions by employing t-SNE visualization

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from matplotlib import rcParams, font_manager
from scipy.stats import gaussian_kde
import os

seed = 80
np.random.seed(seed)


torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)




def t_sne(z_real, z_synthetic, female_indices, male_indices, large_pheno_indices, small_pheno_indices):
    folder = "../figures_experiments/t_sne_approximations/"
    os.makedirs(folder, exist_ok=True)

    ### t-sne of all the dimensions of the posterior
    num_real = z_real.shape[0]

    z_all = np.concatenate((z_real, z_synthetic))

    z_tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=15).fit_transform(z_all)

    z_tsne_real = z_tsne[:num_real, :]
    z_tsne_synthetic = z_tsne[num_real:, :]

    ### Visualize wrt gender
    xmin, xmax = (min(z_tsne_real[female_indices, 0].min(), z_tsne_real[male_indices, 0].min()) - 10,
                  max(z_tsne_real[female_indices, 0].max(), z_tsne_real[male_indices, 0].max()) + 10)

    ymin, ymax = (min(z_tsne_real[female_indices, 1].min(), z_tsne_real[male_indices, 1].min()) - 10,
                  max(z_tsne_real[female_indices, 1].max(), z_tsne_real[male_indices, 1].max()) + 10)

    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 500),
                         np.linspace(ymin, ymax, 500))

    ### Figure 1: real data
    ### Compute the indices
    female_large_pheno_indices = np.intersect1d(female_indices, large_pheno_indices)
    female_small_pheno_indices = np.intersect1d(female_indices, small_pheno_indices)

    male_large_pheno_indices = np.intersect1d(male_indices, large_pheno_indices)
    male_small_pheno_indices = np.intersect1d(male_indices, small_pheno_indices)

    excluded_female = np.union1d(female_large_pheno_indices, female_small_pheno_indices)
    excluded_male = np.union1d(male_large_pheno_indices, male_small_pheno_indices)

    remaining_female = np.setdiff1d(female_indices, excluded_female)
    remaining_male = np.setdiff1d(male_indices, excluded_male)

    plt.figure(figsize=(10, 7))

    plt.scatter(z_tsne[female_small_pheno_indices, 0], z_tsne[female_small_pheno_indices, 1],
                color='purple', marker='v', label='female real', alpha=0.85)

    plt.scatter(z_tsne[female_large_pheno_indices, 0], z_tsne[female_large_pheno_indices, 1],
                color='teal', marker='^', label='female real', alpha=0.85)

    plt.scatter(z_tsne[male_small_pheno_indices, 0], z_tsne[male_small_pheno_indices, 1],
                color='purple', marker='v', label='male real', alpha=0.85)

    plt.scatter(z_tsne[male_large_pheno_indices, 0], z_tsne[male_large_pheno_indices, 1],
                color='teal', marker='^', label='male real', alpha=0.85)

    plt.scatter(z_tsne[remaining_female, 0], z_tsne[remaining_female, 1],
                color='purple', marker='o', alpha=0.25)

    plt.scatter(z_tsne[remaining_male, 0], z_tsne[remaining_male, 1],
                color='teal', marker='o', alpha=0.25)
    # KDE for females
    f_kde = gaussian_kde(np.vstack([z_tsne_real[female_indices, 0], z_tsne_real[female_indices, 1]]))
    f_z = f_kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    plt.contour(xx, yy, f_z, colors='purple', levels=7, linewidths=1.6, alpha=0.9)

    # KDE for males
    m_kde = gaussian_kde(np.vstack([z_tsne_real[male_indices, 0], z_tsne_real[male_indices, 1]]))
    m_z = m_kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    plt.contour(xx, yy, m_z, colors='teal', levels=7, linewidths=1.6, alpha=0.9)
    plt.savefig(folder + f"ae_posterior.svg")
    plt.close()
    print("saved ae")

    ### Figure 2: Synthetic data
    plt.figure(figsize=(10, 7))
    plt.scatter(z_tsne_synthetic[female_indices, 0], z_tsne_synthetic[female_indices, 1],
                color='purple', marker='o', label='female real', alpha=0.2)
    plt.scatter(z_tsne_synthetic[male_indices, 0], z_tsne_synthetic[male_indices, 1],
                color='teal', marker='o', label='male real', alpha=0.2)
    # KDE for females
    f_kde = gaussian_kde(np.vstack([z_tsne_synthetic[female_indices, 0], z_tsne_synthetic[female_indices, 1]]))
    f_z = f_kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    plt.contour(xx, yy, f_z, colors='purple', levels=7, linewidths=1.6)

    # KDE for males
    m_kde = gaussian_kde(np.vstack([z_tsne_synthetic[male_indices, 0], z_tsne_synthetic[male_indices, 1]]))
    m_z = m_kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    plt.contour(xx, yy, m_z, colors='teal', levels=7, linewidths=1.6)
    plt.savefig(folder + f"nf_posterior.svg")
    plt.close()
    print("saved nf")


### Figure formatting
params = {
   'axes.labelsize': 27,
   'font.size': 27,
   'legend.fontsize': 27,
   'xtick.labelsize': 27,
   'ytick.labelsize': 27,
   'text.usetex': False,
   'axes.linewidth': 3.1,
   'xtick.major.width': 3.1,
   'ytick.major.width': 3.1,
   'xtick.major.size': 3.1,
   'ytick.major.size': 3.1,
}
plt.rcParams.update(params)
font_path = r'C:\Users\kkevopoulos\AppData\Local\Microsoft\Windows\Fonts\SourceSansPro-Regular.otf'
font_prop = font_manager.FontProperties(fname=font_path)
rcParams['font.family'] = font_prop.get_name()


device = 'cuda'

scaler = MinMaxScaler()

### Reference distributions from the AE
z_latent_train = np.load("../../data_models_saved/data/X_train_z.npy")
z_latent_valid = np.load("../../data_models_saved/data/X_valid_z.npy")
z_latent_test = np.load("../../data_models_saved/data/X_test_z.npy")



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

outliers = np.load('../../utils/outliers_indices.npy')
mask = np.ones(x_confounders.shape[0], dtype=bool)
mask[outliers] = False

x_confounders = x_confounders[mask]

male_indices = np.where(x_confounders[:, 2].flatten() == 0)[0]
female_indices = np.where(x_confounders[:, 2].flatten() == 1)[0]

clinical = pd.read_pickle('../../data_models_saved/data/dataframes_clinical_info/df_real_clinical.pkl')

large_lv_indices = np.array(clinical.loc[clinical['LV_Vol_mL'] > 170, 'Index'])
small_lv_indices = np.array(clinical.loc[clinical['LV_Vol_mL'] < 110, 'Index'])


NumTrainSamples = z_latent_train.shape[0]
NumValidSamples = z_latent_valid.shape[0]


X_train_confounders = scaler.fit_transform(x_confounders[:NumTrainSamples, :])
X_valid_confounders = scaler.transform(x_confounders[NumTrainSamples:NumTrainSamples+NumValidSamples, :])
X_test_confounders = scaler.transform(x_confounders[NumTrainSamples+NumValidSamples:])

X_train_confounders = torch.tensor(X_train_confounders, dtype=torch.float32).to(device)
X_valid_confounders = torch.tensor(X_valid_confounders, dtype=torch.float32).to(device)
X_test_confounders = torch.tensor(X_test_confounders, dtype=torch.float32).to(device)


### Load cNF model to compute the learned distribution
normalizing_flow = torch.load("../../data_models_saved/models/cnf_model.pth", weights_only=False)
normalizing_flow.eval()

z_approx_train, _ = normalizing_flow.reverse(X_train_confounders)
z_approx_valid, _ = normalizing_flow.reverse(X_valid_confounders)
z_approx_test,  _ = normalizing_flow.reverse(X_test_confounders)

z_latent_all = np.concatenate((z_latent_train, z_latent_valid, z_latent_test), axis=0)

z_approx_all = torch.cat((z_approx_train, z_approx_valid, z_approx_test), dim=0)
z_approx_all = z_approx_all.detach().cpu().numpy()


t_sne(z_real=z_latent_all, z_synthetic=z_approx_all, female_indices=female_indices, male_indices=male_indices,
      large_pheno_indices=large_lv_indices, small_pheno_indices=small_lv_indices)

