### This file performs the PCA experiment, to compare the approximation ability of the generative models under consideration
### We perform PCA on the space of momenta --> not the space of reduced representations
### For each generative model, we obtain the corresponding PCA reduced basis coefficients

### We measure the Wasserstein distances between the distributions PCA coefficients of real and synthetic momenta
### We compare this metric for all considered generative models

### For the generative models, we load the synthetic momenta that we generated in the "generate_momenta_general.py" file
### To avoid sampling and generating momenta again


import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
from scipy.stats import gaussian_kde
import os
from scipy.stats import wasserstein_distance
from sklearn.utils.extmath import randomized_svd


def compute_mean_std(results):
    mean = np.mean(results, axis=0)
    std = np.std(results, axis=0)

    return mean, std


def visualize_1d_pca(real_rb, cnf_rb, cvae_rb, num_toplot):

    for i in range(num_toplot):
        reference = real_rb[i]
        synthetic_nf = cnf_rb[i]
        synthetic_vae = cvae_rb[i]

        kde_ref = gaussian_kde(reference)
        x_ref = np.linspace(min(reference), max(reference), reference.shape[0])
        d_ref = kde_ref(x_ref)

        kde_nf = gaussian_kde(synthetic_nf)
        x_nf = np.linspace(min(synthetic_nf), max(synthetic_nf), synthetic_nf.shape[0])
        d_nf = kde_nf(x_nf)

        kde_vae = gaussian_kde(synthetic_vae)
        x_vae = np.linspace(min(synthetic_vae), max(synthetic_vae), synthetic_vae.shape[0])
        d_vae = kde_vae(x_vae)

        plt.figure()
        plt.plot(x_vae, d_vae, color="#CC79A7")
        plt.fill_between(x_vae, d_vae, color="#CC79A7", alpha=0.15)
        plt.plot(x_ref, d_ref, color='black')
        plt.fill_between(x_ref, d_ref, color='black', alpha=0.15)
        plt.plot(x_nf, d_nf, color="#56B4E9")
        plt.fill_between(x_nf, d_nf, color="#56B4E9", alpha=0.15)
        # plt.gca().yaxis.set_visible(False)
        # plt.gca().spines['top'].set_visible(False)
        # plt.gca().spines['right'].set_visible(False)
        # plt.gca().spines['left'].set_visible(False)
        plt.savefig(path_1d_dists + f"/pca_mode_{i}.svg")
        plt.close()


def load_momenta(path_init, num_momenta_samples):

    momenta_all = np.empty((num_momenta_samples, 720, 3))
    for i in range(num_momenta_samples):
        path = path_init + f"\Shooting_Momenta_{i}\data"

        momenta = np.loadtxt(path+"\Momenta.txt")
        momenta = np.delete(momenta, 0, axis=0)
        momenta = momenta.reshape((1, 720, 3))

        momenta_all[i] = momenta

    return momenta_all


def SVD(num_components, data_matrix):
    print("Initializing SVD for data matrix")

    U, s, Vh = np.linalg.svd(data_matrix)
    U_red = U[:, :num_components]

    s_selected = s[:num_components]

    system_energy = np.sum(s_selected) / np.sum(s)

    print(f"SVD completed with {system_energy * 100} % of the system energy explained")
    print(f"The number of singular values used (truncation threshold) is {num_components}")
    print(f"SVD for data matrix finished")

    rank_r_SVD_error = np.sum(s[num_components:] ** 2) / np.linalg.norm(data_matrix, ord='fro') ** 2
    SVD_projection_error = 1 - np.sum(s[:num_components] ** 2) / np.sum(s ** 2)

    print(f"The relative rank-r SVD approximation error is {rank_r_SVD_error * 100}%")
    print(
        f"The relative error of SVD projection is {SVD_projection_error * 100}%\nThis is "
        f"the cumulative energy not captured by the projection")

    return U, U_red


def PCA(reference_group, generated_group, num_components, random_svd=False):

    ### Perform PCA on the reference group
    X_mean_ref = np.mean(reference_group, axis=1)
    X_ref = reference_group - X_mean_ref[:, np.newaxis]

    X_mean_gen = np.mean(generated_group, axis=1)
    X_gen = generated_group - X_mean_gen[:, np.newaxis]

    if not random_svd:
        _, U_red_reference = SVD(num_components=num_components, data_matrix=X_ref)
    else:
        U_red_reference, _, _ = randomized_svd(X_ref, n_components=num_components, random_state=0)

    ### Obtain PCA reduced basis coefficients for both groups
    X_rb_reference = U_red_reference.T @ X_ref
    X_rb_generated = U_red_reference.T @ X_gen

    return X_rb_reference, X_rb_generated



seed = 44
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

params = {
            'axes.labelsize': 20,
            'font.size': 20,
            'legend.fontsize': 20,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'text.usetex': False,
            'axes.linewidth': 2,
            'xtick.major.width': 3,
            'ytick.major.width': 3,
            'xtick.major.size': 3,
            'ytick.major.size': 3
        }
plt.rcParams.update(params)
font_path = r'C:\Users\kkevopoulos\AppData\Local\Microsoft\Windows\Fonts\SourceSansPro-Regular.otf'
font_prop = font_manager.FontProperties(fname=font_path)
rcParams['font.family'] = font_prop.get_name()


path_figures = "../figures_experiments"

path_pca_experiment = path_figures+"/pca_experiment"
os.makedirs(path_pca_experiment, exist_ok=True)

path_1d_dists = path_pca_experiment+"/1d_dists"
os.makedirs(path_1d_dists, exist_ok=True)

path_real_variability = path_pca_experiment+"/real_variability"
os.makedirs(path_1d_dists, exist_ok=True)


scaler = MinMaxScaler()


# ### Real (reference) momenta
momenta = np.loadtxt("../data_models_saved/data/DeterministicAtlas__EstimatedParameters__Momenta.txt")
momenta = np.delete(momenta, 0, axis=0)
momenta = momenta.reshape((2274, 720, 3))
momenta_reference = momenta.transpose(1, 2, 0).reshape(720 * 3, 2274)

N_half = int(momenta_reference.shape[1] / 2)

### Load synthetic momenta from generative models
path_generated = r"C:\Users\kkevopoulos\Documents\Meshes_Anatomies"


### First experiment
### For ALL generative models, we perform the PCA experiment, measure WDs and visualize them
### For the VAEs, we follow the ordering of "generate_momenta_general.py" file
models = ['nf', 'vae2', 'vae3']


### For the real variability --> Split the real dataset randomly 5 times, to account for the splitting stochasticity
num_splits = 5
num_components_pca = 650


wd_allruns_dict = {}
models_dists_rb = []
for i in range(num_splits):
    indices = torch.randperm(momenta_reference.shape[1])

    X1_idx = indices[:N_half]
    X2_idx = indices[N_half:]

    X1_ref_momenta = momenta_reference[:, X1_idx]
    X2_ref_momenta = momenta_reference[:, X2_idx]

    ### First step: PCA between two real subsets
    x1_rb, x2_rb = PCA(reference_group=X1_ref_momenta, generated_group=X2_ref_momenta,
                       num_components=num_components_pca)

    reference_dists_rb = x1_rb

    wd_real_list = []
    for k in range(num_components_pca):
        wd_real = wasserstein_distance(x1_rb[k], x2_rb[k])
        wd_real_list.append(wd_real)


    wd_models_list = []
    for model in models:
        path_female = path_generated + "\Female_gen_" + model + "_Momenta"
        path_male = path_generated + "\Male_gen_" + model + "_Momenta"

        momenta_female = load_momenta(path_init=path_female, num_momenta_samples=300)
        momenta_male = load_momenta(path_init=path_male, num_momenta_samples=300)

        momenta_generated_all = np.concatenate((momenta_female, momenta_male), axis=0)
        momenta_generated_all = momenta_generated_all.transpose(1, 2, 0)
        momenta_generated_all = momenta_generated_all.reshape(720 * 3, -1)

        ### PCA between real subset X1 and synthetic momenta
        x_rb_real, x_rb_generated = PCA(reference_group=X1_ref_momenta, generated_group=momenta_generated_all,
                                        num_components=num_components_pca)

        if i == 0:
            models_dists_rb.append(x_rb_generated)

        wd_synthetic_list = []
        for k in range(num_components_pca):
            wd_synthetic_model = wasserstein_distance(x_rb_real[k], x_rb_generated[k])
            wd_synthetic_list.append(wd_synthetic_model)

        wd_models_list.append(wd_synthetic_list)
    wd_models_list.append(wd_real_list)

    wd_allruns_dict[f"run_{i}"] = wd_models_list


cnf_results = np.stack([wd_allruns_dict[f"run_{i}"][0] for i in range(num_splits)])
cvae2_results = np.stack([wd_allruns_dict[f"run_{i}"][1] for i in range(num_splits)])
cvae3_results = np.stack([wd_allruns_dict[f"run_{i}"][2] for i in range(num_splits)])
real_results = np.stack([wd_allruns_dict[f"run_{i}"][3] for i in range(num_splits)])

### Compute mean and std to visualize
cnf_mean, cnf_std = compute_mean_std(cnf_results)
cvae2_mean, cvae2_std = compute_mean_std(cvae2_results)
cvae3_mean, cvae3_std = compute_mean_std(cvae3_results)
real_mean, real_std = compute_mean_std(real_results)

results = [[cnf_mean, cnf_std],
           [cvae2_mean, cvae2_std],
           [cvae3_mean, cvae3_std]]




### Variability experiment
plt.figure(figsize=(7, 6))
for i in range(len(results)):

    if i == 0:
        color = "#56B4E9"
        label = 'CAN-DO'
    elif i == 1:
        color = 'orange'
        label = r'$\beta=10^{-2}$'
    elif i == 2:
        color = "#CC79A7"
        label = r'$\beta=10^{-3}$'

    mean = np.array(real_mean) / np.array(results[i][0])
    minus_std = np.array(real_mean - real_std) / np.array(results[i][0]-results[i][1])
    plus_std = np.array(real_mean + real_std) / np.array(results[i][0]+results[i][1])

    line, = plt.plot(mean, color=color, label=label, linewidth=2)
    line.set_clip_on(True)
    plt.fill_between(np.arange(cnf_results.shape[1]), minus_std, plus_std, color=color, alpha=0.25, clip_on=True, rasterized=True)



plt.axhline(y=1, color='black', linestyle='--')
plt.yscale("log")
plt.xscale("log")
plt.xlabel('PCA coefficients')
# plt.ylabel(r"$\frac{WD(\mathbf{X}_1, \mathbf{X}_2)}{WD(\mathbf{X}_1, \tilde{\mathbf{X}}_{\text{gen}})}$")
plt.savefig('../figures_experiments/pca_experiment/variability_exp.svg')
plt.close()





### Figure 2 --- 1D distributions of PCA reduced basis coefficients of momenta, for different generative models
visualize_1d_pca(real_rb=reference_dists_rb, cnf_rb=models_dists_rb[0],
                 cvae_rb=models_dists_rb[2], num_toplot=10)





### Figure 3 --- Variability experiment with respect to sex
x_confounders = pd.read_excel(r"C:\Users\kkevopoulos\OneDrive - Delft University of Technology\Bureaublad\data_kostas_bivme\metadata_final.xlsx")
x_confounders.drop(['Participant ID', 'Height', 'Weight', 'Diastolic BP',
                    'Systolic BP', 'Unnamed: 8', 'Unnamed: 9', 'subject_id'], axis=1, inplace=True)
x_confounders = pd.get_dummies(x_confounders, columns=['Sex'])
x_confounders['Sex_Female'] = x_confounders['Sex_Female'].replace({True: 1, False: 0})
x_confounders['Sex_Male'] = x_confounders['Sex_Male'].replace({True: 1, False: 0})
x_confounders = x_confounders[['BMI', 'Age', 'Sex_Female', 'Sex_Male']]
x_confounders = x_confounders.to_numpy()

sex_info = x_confounders[:, 2]
male_indices = np.where(sex_info.flatten() == 0)[0]
female_indices = np.where(sex_info.flatten() == 1)[0]

real_momenta_male = momenta_reference[:, male_indices]
real_momenta_female = momenta_reference[:, female_indices]

x1r_rb, x2r_rb = PCA(reference_group=real_momenta_male, generated_group=real_momenta_female,
                       num_components=num_components_pca)

wd_real_list = []
for k in range(num_components_pca):
    wd = wasserstein_distance(x1r_rb[k], x2r_rb[k])
    wd_real_list.append(wd)

models = ['nf', 'vae2', 'vae3']
wd_gen_lists_all = []
for model in models:
    path_female = path_generated + "\Female_gen_" + model + "_Momenta"
    path_male = path_generated + "\Male_gen_" + model + "_Momenta"

    momenta_female = load_momenta(path_init=path_female, num_momenta_samples=300)
    momenta_female = momenta_female.transpose(1, 2, 0)
    momenta_female = momenta_female.reshape(720 * 3, -1)


    momenta_male = load_momenta(path_init=path_male, num_momenta_samples=300)
    momenta_male = momenta_male.transpose(1, 2, 0)
    momenta_male = momenta_male.reshape(720 * 3, -1)


    ### PCA between real subset X1 and synthetic momenta
    x1g_rb, x2g_rb = PCA(reference_group=real_momenta_male, generated_group=momenta_female,
                                    num_components=num_components_pca)

    wd_gen_list = []
    for k in range(num_components_pca):
        wd = wasserstein_distance(x1g_rb[k], x2g_rb[k])
        wd_gen_list.append(wd)

    wd_gen_lists_all.append(wd_gen_list)


plt.figure(figsize=(7, 6))
plt.plot(np.array(wd_real_list) / np.array(wd_gen_lists_all[0]), color="#56B4E9", label='CAN-DO', linewidth=2)
plt.axhline(y=1, color='black', linestyle='--')
plt.plot(np.array(wd_real_list) / np.array(wd_gen_lists_all[1]), color='orange', label=r'cVAE $\beta=10^{-2}$', linewidth=2)
plt.plot(np.array(wd_real_list) / np.array(wd_gen_lists_all[2]), color="#CC79A7", label=r'cVAE $\beta=10^{-3}$', linewidth=2)


plt.yscale("log")
plt.xscale("log")
plt.xlabel('PCA coefficients')
# plt.ylabel(r"$\frac{WD(\mathbf{X}^m, \mathbf{X}^f)}{WD(\mathbf{X}^m, \tilde{\mathbf{X}}^f_{\text{gen}})}$")
plt.savefig('../figures_experiments/pca_experiment/variability_exp_sex.svg')


