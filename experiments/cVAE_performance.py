import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from matplotlib import rcParams, font_manager
from scipy.stats import gaussian_kde
import ot


def Wasserstein_distance(X, Y):
    "Entropically regularized optimal transport cost between two empirical distributions"
    M = ot.dist(X, Y, metric='euclidean')
    a = np.ones(len(X)) / len(X)
    b = np.ones(len(Y)) / len(Y)

    W = ot.sinkhorn2(a, b, M, reg=2.1)

    return W


def reparametrize(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    return mean + eps * std


def t_sne(z, female_indices, male_indices, mod_str):
    folder = "../figures_experiments/t_sne_approximations/cvae_posteriors/"
    os.makedirs(folder, exist_ok=True)

    z_tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=15).fit_transform(z)
    print('tsne ok')

    ### Visualize wrt gender
    xmin, xmax = (min(z_tsne[female_indices, 0].min(), z_tsne[male_indices, 0].min()) - 10,
                  max(z_tsne[female_indices, 0].max(), z_tsne[male_indices, 0].max()) + 10)

    ymin, ymax = (min(z_tsne[female_indices, 1].min(), z_tsne[male_indices, 1].min()) - 10,
                  max(z_tsne[female_indices, 1].max(), z_tsne[male_indices, 1].max()) + 10)

    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 500),
                         np.linspace(ymin, ymax, 500))

    plt.figure(figsize=(10, 7))
    plt.scatter(z_tsne[female_indices, 0], z_tsne[female_indices, 1],
                color='purple', marker='o', label='female real', alpha=0.15)
    plt.scatter(z_tsne[male_indices, 0], z_tsne[male_indices, 1],
                color='teal', marker='o', label='male real', alpha=0.15)
    # KDE for females
    f_kde = gaussian_kde(np.vstack([z_tsne[female_indices, 0], z_tsne[female_indices, 1]]))
    f_z = f_kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    plt.contour(xx, yy, f_z, colors='purple', levels=7, linewidths=1.6)

    # KDE for males
    m_kde = gaussian_kde(np.vstack([z_tsne[male_indices, 0], z_tsne[male_indices, 1]]))
    m_z = m_kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    plt.contour(xx, yy, m_z, colors='teal', levels=7, linewidths=1.6)
    plt.savefig(folder + f"{mod_str}_posterior.svg")
    plt.close()

params = {
   'axes.labelsize': 20,
   'font.size': 20,
   'legend.fontsize': 20,
   'xtick.labelsize': 20,
   'ytick.labelsize': 20,
   'text.usetex': False,
   'axes.linewidth': 2.5,
   'xtick.major.width': 2.5,
   'ytick.major.width': 2.5,
   'xtick.major.size': 2.5,
   'ytick.major.size': 2.5,
}
plt.rcParams.update(params)
font_path = r'C:\Users\kkevopoulos\AppData\Local\Microsoft\Windows\Fonts\SourceSansPro-Regular.otf'
font_prop = font_manager.FontProperties(fname=font_path)
rcParams['font.family'] = font_prop.get_name()

device = 'cuda'
scaler = MinMaxScaler()

### Load the metadata
x_confounders = pd.read_excel(r"C:\Users\kkevopoulos\OneDrive - Delft University of Technology\Bureaublad\data_kostas_bivme\metadata_final.xlsx")
x_confounders.drop(['Participant ID', 'Height', 'Weight', 'Diastolic BP',
                    'Systolic BP', 'Unnamed: 8', 'Unnamed: 9', 'subject_id'], axis=1, inplace=True)
x_confounders = pd.get_dummies(x_confounders, columns=['Sex'])
x_confounders['Sex_Female'] = x_confounders['Sex_Female'].replace({True: 1, False: 0})
x_confounders['Sex_Male'] = x_confounders['Sex_Male'].replace({True: 1, False: 0})
x_confounders = x_confounders[['BMI', 'Age', 'Sex_Female', 'Sex_Male']]
x_confounders = x_confounders.to_numpy()

male_indices = np.where(x_confounders[:, 2].flatten() == 0)[0]
female_indices = np.where(x_confounders[:, 2].flatten() == 1)[0]

frac_train = 0.7
frac_valid = 0.15

NumAll = 2274
NumTrainSamples = int(frac_train * NumAll)
NumValidSamples = int(frac_valid * NumAll)

X_train_confounders = scaler.fit_transform(x_confounders[:NumTrainSamples, :])
X_valid_confounders = scaler.transform(x_confounders[NumTrainSamples:NumTrainSamples+NumValidSamples, :])
X_test_confounders = scaler.transform(x_confounders[NumTrainSamples+NumValidSamples:])

X_train_confounders = torch.tensor(X_train_confounders, dtype=torch.float32).to(device)
X_valid_confounders = torch.tensor(X_valid_confounders, dtype=torch.float32).to(device)
X_test_confounders = torch.tensor(X_test_confounders, dtype=torch.float32).to(device)

X_all_confounders = torch.cat((X_train_confounders, X_valid_confounders, X_test_confounders), dim=0)

### Load momenta
momenta = np.loadtxt("../data_models_saved/data/DeterministicAtlas__EstimatedParameters__Momenta.txt")
momenta = np.delete(momenta, 0, axis=0)
momenta = momenta.reshape((NumAll, 720, 3))
momenta = momenta.reshape((NumAll, 3, 8, 9, 10))
momenta = torch.tensor(momenta, dtype=torch.float32, device=device)

cvae1 = torch.load("../data_models_saved/models/cvae_encoder_beta_0.1.pth", weights_only=False)
cvae2 = torch.load("../data_models_saved/models/cvae_encoder_beta_0.01.pth", weights_only=False)
cvae3 = torch.load("../data_models_saved/models/cvae_encoder_beta_0.001.pth", weights_only=False)
cvae4 = torch.load("../data_models_saved/models/cvae_encoder_beta_0.0001.pth", weights_only=False)
cvae5 = torch.load("../data_models_saved/models/cvae_encoder_beta_1e-05.pth", weights_only=False)
cvae6 = torch.load("../data_models_saved/models/cvae_encoder_beta_1e-06.pth", weights_only=False)

cvae1.eval()
cvae2.eval()
cvae3.eval()
cvae4.eval()
cvae5.eval()
cvae6.eval()

models_str = ['vae1', 'vae2', 'vae3', 'vae4', 'vae5', 'vae6']
models = [cvae1, cvae2, cvae3, cvae4, cvae5, cvae6]

latent_dimension = 44
gaussian_prior = torch.distributions.MultivariateNormal(torch.zeros(latent_dimension), torch.eye(latent_dimension))

samples_gaussian_prior = gaussian_prior.sample((NumAll,))
samples_gaussian_prior = samples_gaussian_prior.detach().numpy()

wd_all_vals = []
wd_male_female_vals = []
for str, mod in zip(models_str, models):

    ### Compute Wasserstein distance between latent distribution and gaussian
    mu, logvar = mod(momenta, X_all_confounders)
    z_latent_all = reparametrize(mu, logvar)
    z_latent_all = z_latent_all.detach().cpu().numpy()

    wd = Wasserstein_distance(z_latent_all, samples_gaussian_prior)
    wd_all_vals.append(wd)

    ### Compute Wasserstein distance between female and male latent distributions
    z_latent_female = z_latent_all[female_indices]
    z_latent_male = z_latent_all[male_indices]

    wd_sex = Wasserstein_distance(z_latent_female, z_latent_male)
    wd_male_female_vals.append(wd_sex)

    t_sne(z=z_latent_all, male_indices=male_indices, female_indices=female_indices, mod_str=str)


### Compute Wasserstein distance between latent distribution and gaussian for real/AE distribution
z_latent_train = np.load("../data_models_saved/data/X_train_z.npy")
z_latent_valid = np.load("../data_models_saved/data/X_valid_z.npy")
z_latent_test = np.load("../data_models_saved/data/X_test_z.npy")

z_latent_all_real = np.concatenate((z_latent_train, z_latent_valid, z_latent_test), axis=0)
wd_real = Wasserstein_distance(X=z_latent_all_real, Y=samples_gaussian_prior)

### Compute Wasserstein distance between female and male for real/AE latent distribution
z_latent_female_real = z_latent_all_real[female_indices]
z_latent_male_real = z_latent_all_real[male_indices]
wd_sex_real = Wasserstein_distance(X=z_latent_female_real, Y=z_latent_male_real)



categories = [r'$10^{-1}$', r'$10^{-2}$', r'$10^{-3}$', r'$10^{-4}$', r'$10^{-5}$', r'$10^{-6}$', 'AE']

### Visualize WD between latent distributions and gaussian
folder = "../figures_experiments/t_sne_approximations/cvae_posteriors/"

wd_all_vals.append(wd_real)
wd_male_female_vals.append(wd_sex_real)
x = np.arange(len(categories))
width = 0.35

plt.figure(figsize=(6.7, 5.5))
plt.bar(x - width/2, wd_all_vals, width, color='darkred', label=r'$z \text{ Vs } \mathcal{N}(\mathbf{0}, \mathbf{I})$', clip_on=True, rasterized=True)
plt.bar(x + width/2, wd_male_female_vals, width, color='navy',label=r'$z_{female} \text{ Vs } z_{male}$', clip_on=True, rasterized=True)
plt.xticks(x, categories)
plt.xlabel("models")
plt.ylabel("OT cost")
plt.legend()
plt.yscale('log')
plt.savefig(folder + f"bar_ot_costs.svg")
plt.close()