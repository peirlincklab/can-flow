import numpy as np
from nf_ae_experiments_complete import Compare_shape_dists
import os
from scipy.stats import qmc
from sklearn.preprocessing import minmax_scale
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch


scaler = MinMaxScaler()

def LatinHypercube(dim_sample, low_bounds, upp_bounds, num_samples):
    """
    Function that is used to sample the parameters from a latin hypercube.
    :param dim_sample: The dimension that we sample
    :param low_bounds: lower bound of the sampling interval
    :param upp_bounds: upper bound of the sampling interval
    :param num_samples: number of desired samples
    :return:
    """
    sampler = qmc.LatinHypercube(d=dim_sample)
    sample = sampler.random(n=num_samples)

    l_bounds = low_bounds
    u_bounds = upp_bounds
    sample_params = qmc.scale(sample, l_bounds, u_bounds)
    return sample_params


def load_cp_momenta(file_name):

    # Get the path to the Downloads folder dynamically
    downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")

    file_path = os.path.join(downloads_folder, file_name)
    data = np.loadtxt(file_path)

    return data


def generate_momenta(num_generated, gender, x_conf):
    x_confounders = scaler.inverse_transform(x_conf)
    if gender == "Female":
        sex1 = 1
        sex2 = 0
    else:
        sex1 = 0
        sex2 = 1

    sex1_all = np.tile(sex1, num_generated).reshape(-1, 1)
    sex2_all = np.tile(sex2, num_generated).reshape(-1, 1)

    generated_sex = np.concatenate((sex1_all, sex2_all), axis=1)

    indices = np.where(x_confounders[:, 2] == sex1)[0]

    min_bmi = np.min(x_confounders[indices][:, 0])
    max_bmi = np.max(x_confounders[indices][:, 0])
    min_age = np.min(x_confounders[indices][:, 1])
    max_age = np.max(x_confounders[indices][:, 1])

    generated_bmi = np.random.uniform(low=min_bmi, high=max_bmi, size=num_generated).reshape(-1, 1)
    generated_bmi = minmax_scale(generated_bmi, axis=0)

    generated_age = np.random.randint(low=min_age, high=max_age, size=num_generated).reshape(-1, 1)
    generated_age = minmax_scale(generated_age, axis=0)

    generated_conf_samples = np.concatenate((generated_sex, generated_bmi, generated_age), axis=1)
    generated_conf_samples = torch.tensor(generated_conf_samples, dtype=torch.float32).to(device)

    z_synthetic = norm_flow.reverse(generated_conf_samples)
    generated_shapes = decoder(z_synthetic)

    generated_shapes = generated_shapes.reshape(num_generated, 2730, 3)
    generated_shapes = generated_shapes.detach().cpu().numpy()
    generated_shapes = generated_shapes.transpose(1, 2, 0).reshape(2730 * 3, num_generated)

    return generated_shapes


file_path_confounders = r'C:\Users\kkevopoulos\Downloads\cardiac_function_mesh_complete_final.xlsx'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

latent_dimension = 50


x_confounders = pd.read_excel(file_path_confounders)
x_confounders.drop(['BSA', 'Year of birth', 'Height', 'Weight', 'Diastolic BP mean reading',
                 'Systolic BP mean reading', 'mesh_file', 'MAP'], axis=1, inplace=True)
x_confounders = pd.get_dummies(x_confounders, columns=['Sex'])
x_confounders['Sex_Female'] = x_confounders['Sex_Female'].replace({True: 1, False: 0})
x_confounders['Sex_Male'] = x_confounders['Sex_Male'].replace({True: 1, False: 0})
x_confounders = x_confounders.to_numpy()
x_confounders = x_confounders[:, 1:]
x_confounders = scaler.fit_transform(x_confounders)

file_path_confounders = r'C:\Users\kkevopoulos\Downloads\cardiac_function_mesh_complete_final.xlsx'

file_name_momenta = r"cardiac_atlas_kostas\output\DeterministicAtlas__EstimatedParameters__Momenta.txt"
momenta = load_cp_momenta(file_name_momenta)

momenta = np.delete(momenta, 0, axis=0)
momenta = momenta.reshape((456, 2730, 3))
momenta = momenta.transpose(1, 2, 0).reshape(2730 * 3, 456)

sex_info = x_confounders[:, 2]
male_indices = np.where(sex_info.flatten() == 0)[0]
female_indices = np.where(sex_info.flatten() == 1)[0]



decoder = torch.load("../data_models_saved/decoder.pth")
norm_flow = torch.load("../data_models_saved/flow_model.pth")

decoder.eval()
norm_flow.eval()

########################################################################################################################
###### EXPERIMENT 1:
###### Split available data in two parts. Wasserstein distance of the Real V Real distributions

###### Generate 2000 generated data, for different metadata information (males & females)
###### Split this generated set in 2. Wasserstein distance of Gen V Gen distributions

###### Generate 700 data, and combine them with 456 real data.
###### Wasserstein distance of Gen V Real data
#########################################################################################################################

# ### Real momenta
real_momenta_1 = momenta[:, int(456/2):]
real_momenta_2 = momenta[:, :int(456/2)]

diff = Compare_shape_dists(reference_group=real_momenta_1, test_group=real_momenta_2)
w_diff_real = diff.PCA_Experiment(custom_title='real V real', num_components=450, standardize=False, rand_svd=False)



### Generate 2000 momenta (mixed male and female)
generated_male = generate_momenta(num_generated=1000, gender="Male", x_conf=x_confounders)
generated_female = generate_momenta(num_generated=1000, gender="Female", x_conf=x_confounders)

rng = np.random.default_rng()

generated_all = np.concatenate((generated_male, generated_female), axis=1)
generated_all = rng.permutation(generated_all, axis=1)

generated_group1 = generated_all[:, :1000]
generated_group2 = generated_all[:, 1000:]

diff = Compare_shape_dists(reference_group=generated_group1, test_group=generated_group2)
w_diff_gen = diff.PCA_Experiment(custom_title='gen V gen', rand_svd=True, num_components=1000, standardize=False)


### Compare real V generated (mixed male and female)
diff = Compare_shape_dists(reference_group=real_momenta_1, test_group=generated_group1)
w_diff_real_gen = diff.PCA_Experiment(custom_title='real V gen', num_components=1000, standardize=False, rand_svd=True)

plt.figure()
plt.semilogy(w_diff_real[:220], color='black', label='Real V Real')
plt.semilogy(w_diff_gen[:220], color='red', label='Gen V Gen')
plt.semilogy(w_diff_real_gen[:220], color='blue', label='Real V Gen')
plt.legend()
plt.xlabel("Reduced bases")
plt.ylabel("Wasserstein distance")
plt.show()


########################################################################################################################
###### EXPERIMENT 2: Gender differences
###### Split available data in two parts (male + female). Wasserstein distance of the Real male V Real female distributions

###### Generate 2000 generated data, (1000 for males and 1000 for females)
###### Wasserstein distance of Gen male V Gen female distributions

###### Generate 700 data, and combine them with 456 real data.
###### Wasserstein distance of Real male V Generated female
#########################################################################################################################

### Real male V Real female
real_momenta_male = momenta[:, male_indices]
real_momenta_female = momenta[:, female_indices]

diff = Compare_shape_dists(reference_group=real_momenta_male, test_group=real_momenta_female)
w_diff_real_male_female = diff.PCA_Experiment(custom_title='real male V real female', rand_svd=True, standardize=False, num_components=220)


### Generated male V Generated female
diff = Compare_shape_dists(reference_group=generated_male, test_group=generated_female)
w_diff_gen_male_female = diff.PCA_Experiment(custom_title='gen male V gen female', rand_svd=True, standardize=False, num_components=220)


### Real male V Generated female
diff = Compare_shape_dists(reference_group=real_momenta_male, test_group=generated_female)
w_diff_real_male_gen_female = diff.PCA_Experiment(custom_title='gen male V gen female', rand_svd=True, standardize=False, num_components=220)

plt.figure()
plt.semilogy(w_diff_real_male_female[:220], color='black', label='Real male V Real female')
plt.semilogy(w_diff_gen_male_female[:220], color='red', label='Gen male V Gen female')
plt.semilogy(w_diff_real_male_gen_female[:220], color='blue', label='Real male V Gen female')
plt.legend()
plt.xlabel("Reduced bases")
plt.ylabel("Wasserstein distance")
plt.show()














