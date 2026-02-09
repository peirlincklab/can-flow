import numpy as np
from sklearn.preprocessing import  MinMaxScaler
import torch
import pandas as pd
from utils import save_write_momenta

### This file generates synthetic momenta, according to metadata from specific population subgroups
### We manually define a population subgroup
### From this group, we randomly sample some metadata vectors
### And then we generate the corresponding momenta using the cNF and cVAE models


def generate_momenta(model_str, model, sampled_metadata):
    latent_dim = 44

    if model_str == 'nf':
        z_synthetic, _ = model.reverse(sampled_metadata)
        generated_shapes = ae_decoder(z_synthetic)

    else:
        latent_dist = torch.distributions.MultivariateNormal(torch.zeros(latent_dim), torch.eye(latent_dim))
        z_latent = latent_dist.sample((sampled_metadata.shape[0],)).to('cuda')
        generated_shapes = model(z_latent, sampled_metadata)

    generated_shapes = generated_shapes.reshape(sampled_metadata.shape[0], 720, 3)
    generated_shapes = generated_shapes.detach().cpu().numpy()

    return generated_shapes


def sample_metadata_targeted(num_samples, gen_age, gen_sex, x_confounders):
    sex0 = gen_sex[0]

    sex1_all = np.tile(gen_sex[0], num_samples).reshape(-1, 1)
    sex2_all = np.tile(gen_sex[1], num_samples).reshape(-1, 1)

    gen_sex = np.concatenate((sex1_all, sex2_all), axis=1)
    gen_age = np.tile(gen_age, num_samples).reshape(-1, 1)

    indices = np.where(x_confounders[:, 2] == sex0)[0]

    min_bmi = np.min(x_confounders[indices][:, 0])
    max_bmi = np.max(x_confounders[indices][:, 0])

    gen_bmi = np.random.uniform(low=min_bmi, high=max_bmi, size=num_samples).reshape(-1, 1)

    metadata_samples = np.concatenate((gen_bmi, gen_age, gen_sex), axis=1)

    return metadata_samples



scaler = MinMaxScaler()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


frac_train = 0.7

NumAll =  2274
NumTrainSamples = int(NumAll * frac_train)


### Load the confounders just to apply the scaler to the training data
x_confounders = pd.read_excel(r"C:\Users\kkevopoulos\OneDrive - Delft University of Technology\Bureaublad\data_kostas_bivme\metadata_final.xlsx")
x_confounders.drop(['Participant ID', 'Height', 'Weight', 'Diastolic BP',
                    'Systolic BP', 'Unnamed: 8', 'Unnamed: 9', 'subject_id'], axis=1, inplace=True)
x_confounders = pd.get_dummies(x_confounders, columns=['Sex'])
x_confounders['Sex_Female'] = x_confounders['Sex_Female'].replace({True: 1, False: 0})
x_confounders['Sex_Male'] = x_confounders['Sex_Male'].replace({True: 1, False: 0})
x_confounders = x_confounders[['BMI', 'Age', 'Sex_Female', 'Sex_Male']]
x_confounders = x_confounders.to_numpy()

x_confounders_train = scaler.fit_transform(x_confounders[:NumTrainSamples, :])

### Define the characteristics of the population subgroup we want to generate anatomies for
age_subgroup = 75
sex_subgroup = [1, 0]

### How many samples to generate for the desired metadata vector
num_samples = 700
targeted_metadata_sampled = sample_metadata_targeted(num_samples=num_samples, gen_age=age_subgroup,
                                                     gen_sex=sex_subgroup, x_confounders=x_confounders)

targeted_metadata_sampled = scaler.transform(targeted_metadata_sampled)
targeted_metadata_sampled = torch.tensor(targeted_metadata_sampled, dtype=torch.float32).to(device)

### Load the cNF and cVAE models to generate synthetic momenta
ae_decoder = torch.load("../data_models_saved/models/ae_decoder.pth", weights_only=False)
cnf = torch.load("../data_models_saved/models/cnf_model.pth", weights_only=False)

cvae1 = torch.load("../data_models_saved/models/cvae_decoder_beta_0.1.pth", weights_only=False)
cvae2 = torch.load("../data_models_saved/models/cvae_decoder_beta_0.01.pth", weights_only=False)
cvae3 = torch.load("../data_models_saved/models/cvae_decoder_beta_0.001.pth", weights_only=False)
cvae4 = torch.load("../data_models_saved/models/cvae_decoder_beta_0.0001.pth", weights_only=False)
cvae5 = torch.load("../data_models_saved/models/cvae_decoder_beta_1e-05.pth", weights_only=False)
cvae6 = torch.load("../data_models_saved/models/cvae_decoder_beta_1e-06.pth", weights_only=False)


ae_decoder.eval()
cnf.eval()

cvae1.eval()
cvae2.eval()
cvae3.eval()
cvae4.eval()
cvae5.eval()
cvae6.eval()


models_str = ['nf', 'vae1', 'vae2', 'vae3', 'vae4', 'vae5', 'vae6']
models = [cnf, cvae1, cvae2, cvae3, cvae4, cvae5, cvae6]


for str, mod in zip(models_str, models):

    generated_momenta_targeted = generate_momenta(model_str=str, model=mod,
                                                  sampled_metadata=targeted_metadata_sampled)

    save_write_momenta.save_momenta(type_momenta="Targeted" + "_" + str,
                                    momenta_tosave=generated_momenta_targeted)
