import numpy as np
from sklearn.preprocessing import  MinMaxScaler
import torch
import pandas as pd
from utils import save_write_momenta

### This file generates synthetic momenta, according to metadata from specific population subgroups
### We manually define a population subgroup
### From this group, we randomly sample some metadata vectors
### And then we generate the corresponding momenta using CAN-FLOW and cVAE models

### The synthetic cohorts are used to produce the results of Figure 8 of the manuscript


def generate_momenta(model_str, model, sampled_metadata):
    """
          Generate synthetic momenta using trained CAN-FLOW or trained cVAEs

          If `model_str` is "nf", CAN-FLOW is used to generate momenta.

          Otherwise, a cVAE is used to generate momenta.

          Parameters
          ----------
          model_str : str
              String identifying the type of generative model. If set to "nf", the
              generative model is CAN-FLOW. Any other value assumes that a cVAE of a specific regularization strength is used.
          model : torch.nn.Module
              Generative model used to produce synthetic momenta.
          sampled_metadata : torch.Tensor
              Metadata used for conditional generation, with shape
              (n_samples, n_metadata_features).

          Returns
          -------
          np.ndarray
              Generated synthetic momenta with shape (n_samples, 720, 3).

          Notes
          -----
          For the non-normalizing-flow case, latent vectors are sampled on the CPU
          and then moved to CUDA. Therefore, this function assumes that a CUDA device
          is available.
          """
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
    """
        Generate synthetic metadata samples for a specified sex group and age range.

        The function creates metadata samples containing BMI, age, and one-hot
        encoded sex information. The sex encoding is fixed according to `gen_sex`.
        BMI values are sampled uniformly from the observed BMI range of the selected
        sex group in `x_confounders`. Age values are sampled uniformly from
        `gen_age` to the maximum observed age of the selected sex group.

        Parameters
        ----------
        num_samples : int
            Number of metadata samples to generate.
        gen_age : float
            Lower bound of the age range used for sampling generated ages.
        gen_sex : array-like
            One-hot encoded sex information with shape (2,). The first entry is
            used to select the corresponding sex group from `x_confounders`.
            For example, `[1, 0]` represents female and `[0, 1]` represents
            male.
        x_confounders : np.ndarray
            Array containing observed metadata with shape
            (n_samples, n_confounders). The expected column order is:

            - `x_confounders[:, 0]`: BMI values
            - `x_confounders[:, 1]`: age values
            - `x_confounders[:, 2]`: first sex indicator

        Returns
        -------
        np.ndarray
            Generated metadata samples with shape (num_samples, 4). The columns
            are ordered as:

            - BMI
            - age
            - first sex indicator
            - second sex indicator
        """
    sex0 = gen_sex[0]

    sex1_all = np.tile(gen_sex[0], num_samples).reshape(-1, 1)
    sex2_all = np.tile(gen_sex[1], num_samples).reshape(-1, 1)

    gen_sex_samples = np.concatenate((sex1_all, sex2_all), axis=1)

    indices = np.where(x_confounders[:, 2] == sex0)[0]

    min_bmi = np.min(x_confounders[indices][:, 0])
    max_bmi = np.max(x_confounders[indices][:, 0])

    min_age = gen_age
    max_age = np.max(x_confounders[indices][:, 1])

    gen_bmi_samples = np.random.uniform(low=min_bmi, high=max_bmi, size=num_samples).reshape(-1, 1)
    gen_age_samples = np.random.uniform(low=min_age, high=max_age, size=num_samples).reshape(-1, 1)

    metadata_samples = np.concatenate((gen_bmi_samples, gen_age_samples, gen_sex_samples), axis=1)

    return metadata_samples



scaler = MinMaxScaler()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


frac_train = 0.7

### Load the confounders just to apply the scaler to the training data
x_confounders = pd.read_excel(r"/home/kevopou1/metadata_final.xlsx")
x_confounders.drop(['Participant ID', 'Height', 'Weight', 'Diastolic BP',
                    'Systolic BP', 'Unnamed: 8', 'Unnamed: 9', 'subject_id'], axis=1, inplace=True)
x_confounders = pd.get_dummies(x_confounders, columns=['Sex'])
x_confounders['Sex_Female'] = x_confounders['Sex_Female'].replace({True: 1, False: 0})
x_confounders['Sex_Male'] = x_confounders['Sex_Male'].replace({True: 1, False: 0})
x_confounders = x_confounders[['BMI', 'Age', 'Sex_Female', 'Sex_Male']]
x_confounders = x_confounders.to_numpy()

### Delete outliers and participants that withdrew from the study
x_confounders = np.delete(x_confounders, [1746, 1831], axis=0)

NumAll = x_confounders.shape[0]
NumTrainSamples = int(NumAll * frac_train)

outliers = np.load('../utils/outliers_indices.npy')
mask = np.ones(x_confounders.shape[0], dtype=bool)
mask[outliers] = False

x_confounders = x_confounders[mask]

x_confounders_train = scaler.fit_transform(x_confounders[:NumTrainSamples, :])

### Define the metadata characteristics of the population subgroup we want to generate anatomies for
age_subgroup = 58
sex_subgroup = [0, 1]

### How many samples to generate for the desired metadata vector
num_samples = 650
targeted_metadata_sampled = sample_metadata_targeted(num_samples=num_samples, gen_age=age_subgroup,
                                                     gen_sex=sex_subgroup,
                                                     x_confounders=x_confounders)

targeted_metadata_sampled = scaler.transform(targeted_metadata_sampled)
targeted_metadata_sampled = torch.tensor(targeted_metadata_sampled, dtype=torch.float32).to(device)

### Load CAN-FLOW and cVAE models to generate synthetic momenta
ae_decoder = torch.load("../data_models_saved/models/ae_decoder.pth", weights_only=False)
cnf = torch.load("../data_models_saved/models/cnf_model.pth", weights_only=False)

cvae2 = torch.load("../data_models_saved/models/cvae_decoder_beta_0.01.pth", weights_only=False)
cvae3 = torch.load("../data_models_saved/models/cvae_decoder_beta_0.001.pth", weights_only=False)



ae_decoder.eval()
cnf.eval()

cvae2.eval()
cvae3.eval()


### For this experiment, we only study synthetic cohorts for CAN-FLOW and the best two cVAEs (\beta=10^{-2} and \beta=10^{-3})
models_str = ['nf', 'vae2', 'vae3']
models = [cnf, cvae2, cvae3]

### generate synthetic momenta for the three generative models
for str, mod in zip(models_str, models):

    generated_momenta_targeted = generate_momenta(model_str=str, model=mod,
                                                  sampled_metadata=targeted_metadata_sampled)

    save_write_momenta.save_momenta(type_momenta="Targeted" + "_" + str,
                                    momenta_tosave=generated_momenta_targeted)
