import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import random
from utils import save_write_momenta


def sample_metadata(num_samples, x_confounders):

    sex = ["Female", "Male"]
    choice = random.choice(sex)

    if choice == "Female":
        sex1 = 1
        sex2 = 0
    else:
        sex1 = 0
        sex2 = 1

    sex1_all = np.tile(sex1, num_samples).reshape(-1, 1)
    sex2_all = np.tile(sex2, num_samples).reshape(-1, 1)

    gen_sex = np.concatenate((sex1_all, sex2_all), axis=1)

    indices = np.where(x_confounders[:, 2] == sex1)[0]

    min_bmi = np.min(x_confounders[indices][:, 0])
    max_bmi = np.max(x_confounders[indices][:, 0])
    min_age = np.min(x_confounders[indices][:, 1])
    max_age = np.max(x_confounders[indices][:, 1])

    gen_bmi = np.random.uniform(low=min_bmi, high=max_bmi, size=num_samples).reshape(-1, 1)
    gen_age = np.random.randint(low=min_age, high=max_age, size=num_samples).reshape(-1, 1)

    metadata_samples = np.concatenate((gen_bmi, gen_age, gen_sex), axis=1)

    return metadata_samples


seed = 42

# PyTorch
torch.manual_seed(seed)

# If using GPU
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

np.random.seed(seed)

random.seed(seed)

device = 'cuda'


x_confounders = pd.read_excel(r"C:\Users\kkevopoulos\OneDrive - Delft University of Technology\Bureaublad\data_kostas_bivme\metadata_final.xlsx")
x_confounders.drop(['Participant ID', 'Height', 'Weight', 'Diastolic BP',
                    'Systolic BP', 'Unnamed: 8', 'Unnamed: 9', 'subject_id'], axis=1, inplace=True)
x_confounders = pd.get_dummies(x_confounders, columns=['Sex'])
x_confounders['Sex_Female'] = x_confounders['Sex_Female'].replace({True: 1, False: 0})
x_confounders['Sex_Male'] = x_confounders['Sex_Male'].replace({True: 1, False: 0})
x_confounders = x_confounders[['BMI', 'Age', 'Sex_Female', 'Sex_Male']]
x_confounders = x_confounders.to_numpy()

scaler = MinMaxScaler()

frac_train = 0.7
NumAll =  2274
NumTrainSamples = int(NumAll * frac_train)

x_confounders_train = scaler.fit_transform(x_confounders[:NumTrainSamples, :])


### Load models
cvae1 = torch.load("../data_models_saved/models/cvae_decoder_beta_0.1.pth", weights_only=False)
cvae2 = torch.load("../data_models_saved/models/cvae_decoder_beta_0.01.pth", weights_only=False)
cvae3 = torch.load("../data_models_saved/models/cvae_decoder_beta_0.001.pth", weights_only=False)
cvae4 = torch.load("../data_models_saved/models/cvae_decoder_beta_0.0001.pth", weights_only=False)
cvae5 = torch.load("../data_models_saved/models/cvae_decoder_beta_1e-05.pth", weights_only=False)
cvae6 = torch.load("../data_models_saved/models/cvae_decoder_beta_1e-06.pth", weights_only=False)

cvae1.eval()
cvae2.eval()
cvae3.eval()
cvae4.eval()
cvae5.eval()
cvae6.eval()


models_str = ['vae1', 'vae2', 'vae3', 'vae4', 'vae5', 'vae6']
models = [cvae1, cvae2, cvae3, cvae4, cvae5, cvae6]

latent_dim = 44
latent_dist = torch.distributions.MultivariateNormal(torch.zeros(latent_dim), torch.eye(latent_dim))

### Setting 1: sample one latent representation "z" --> keep this fixed and vary the metadata
num_gen = 300

z_fixed_sample = latent_dist.sample((1,))
z_fixed_sample = z_fixed_sample.repeat(num_gen, 1)
z_fixed_sample = torch.tensor(z_fixed_sample, dtype=torch.float32, device=device)

x_conf_sampled = sample_metadata(num_samples=num_gen, x_confounders=x_confounders)
x_conf_sampled = scaler.transform(x_conf_sampled)
x_conf_sampled = torch.tensor(x_conf_sampled, dtype=torch.float32, device=device)

for str, model in zip(models_str, models):

    gen_momenta = model(z_fixed_sample, x_conf_sampled)
    gen_momenta = gen_momenta.reshape(num_gen, 720, 3)
    gen_momenta = gen_momenta.detach().cpu().numpy()

    save_write_momenta.save_momenta(type_momenta="VAE_Variability_z_fixed_" + str, momenta_tosave=gen_momenta)

### Setting 2: Sample different latent representations z_i --> sample 1 metadata instance and keep this fixed
z_samples = latent_dist.sample((num_gen,))
z_samples = torch.tensor(z_samples, dtype=torch.float32, device=device)

x_conf_fixed_sample = sample_metadata(num_samples=1, x_confounders=x_confounders)
x_conf_fixed_sample = np.tile(x_conf_fixed_sample, (num_gen, 1))
x_conf_fixed_sample = torch.tensor(x_conf_fixed_sample, dtype=torch.float32, device=device)

for str, model in zip(models_str, models):
    gen_momenta = model(z_samples, x_conf_fixed_sample)
    gen_momenta = gen_momenta.reshape(num_gen, 720, 3)
    gen_momenta = gen_momenta.detach().cpu().numpy()

    save_write_momenta.save_momenta(type_momenta="VAE_Variability_z_varied_" + str, momenta_tosave=gen_momenta)




