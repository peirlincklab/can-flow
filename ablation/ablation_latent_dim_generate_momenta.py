import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
from utils import save_write_momenta


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


scaler = MinMaxScaler()


#x_confounders = pd.read_excel(r"/home/kevopou1/metadata_final.xlsx")
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

outliers = np.load('../utils/outliers_indices.npy')
mask = np.ones(x_confounders.shape[0], dtype=bool)
mask[outliers] = False

x_confounders = x_confounders[mask]

NumAll = x_confounders.shape[0]
frac_train = 0.7
NumTrainSamples = int(NumAll * frac_train)

X_train_confounders = scaler.fit_transform(x_confounders[:NumTrainSamples, :])


metadata_sampled_female = np.load('../data_models_saved/data/metadata_sampled_female.npy')
metadata_sampled_male = np.load('../data_models_saved/data/metadata_sampled_male.npy')

metadata_sampled_all = np.concatenate((metadata_sampled_female, metadata_sampled_male), axis=0)
metadata_sampled_all = scaler.transform(metadata_sampled_all)
metadata_sampled_all = torch.tensor(metadata_sampled_all, dtype=torch.float32, device=device)


latent_dims = [10, 20, 30, 40, 50]
betas = ["0.01", "0.001"]

models = ["can-flow", "cvae"]

for model in models:
    for dim in latent_dims:
        if model == "cvae":
            for beta in betas:
                cvae_decoder = torch.load(f"../ablation/ablation_latent_dimensionality/models_saved/cvae_decoder_beta_{beta}_latent_{dim}.pth", weights_only=False)

                latent_dist = torch.distributions.MultivariateNormal(torch.zeros(dim), torch.eye(dim))
                z_latent = latent_dist.sample((metadata_sampled_all.shape[0],)).to('cuda')
                generated_momenta = cvae_decoder(z_latent, metadata_sampled_all)

                generated_momenta = generated_momenta.reshape(metadata_sampled_all.shape[0], 720, 3)
                generated_momenta = generated_momenta.detach().cpu().numpy()

                save_write_momenta.save_momenta(type_momenta=f"AblationLatentDim_cvae3_latent{dim}", momenta_tosave=generated_momenta)
        else:
            nf = None ### TODO TODO TODO TODO to populate

