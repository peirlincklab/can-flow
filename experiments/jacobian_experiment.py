import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import matplotlib.pyplot as plt


def compute_decoder_jacobian(decoder, z, c=None, ae=False):
    """
    decoder: maps (1, latent_dim) -> (1, C, D, H, W)
    z: tensor of shape (1, latent_dim), requires_grad=False or True

    returns:
        J: tensor of shape (output_dim, latent_dim)
    """
    decoder.eval()

    z = z.detach().clone().requires_grad_(True)

    if ae:
        def f(z_in):
            x = decoder(z_in.reshape(1, -1))
            return x.reshape(-1)
    else:
        def f(z_in):
            x = decoder(z_in.reshape(1, -1), c.reshape(1, 4))
            return x.reshape(-1)

    J = torch.autograd.functional.jacobian(f, z)
    return J

def jacobian_effective_rank(X, decoder, c=None, ae=False):
    eps = 1e-15

    num_all_samples = X.shape[0]

    eff_ranks = []
    pbar = tqdm(total=num_all_samples, desc='Computing Jacobian for all samples...', leave=True)
    for i in range(num_all_samples):

        if c is not None:
            cc = c[i]
        else:
            cc = None

        J = compute_decoder_jacobian(decoder=decoder, z=X[i], c=cc, ae=ae)

        # singular values
        S = torch.linalg.svdvals(J)
        p = S / (S.sum() + eps)
        H = -(p * (p + eps).log()).sum()
        eff = torch.exp(H)
        eff_ranks.append(eff.detach().cpu().numpy())

        pbar.update()
    pbar.close()

    return np.mean(eff_ranks), np.std(eff_ranks)


device = 'cuda'


cando_decoder = torch.load('../data_models_saved/models/ae_decoder.pth')
cando_cnf = torch.load("../data_models_saved/models/cnf_model.pth")

cvae1 = torch.load('../data_models_saved/models/cvae_decoder_beta_0.1.pth')
cvae2 = torch.load('../data_models_saved/models/cvae_decoder_beta_0.01.pth')
cvae3 = torch.load('../data_models_saved/models/cvae_decoder_beta_0.001.pth')
cvae4 =  torch.load('../data_models_saved/models/cvae_decoder_beta_0.0001.pth')
cvae5 = torch.load('../data_models_saved/models/cvae_decoder_beta_1e-05.pth')
cvae6 = torch.load('../data_models_saved/models/cvae_decoder_beta_1e-06.pth')

cando_decoder.eval()
cando_cnf.eval()

cvae1.eval()
cvae2.eval()
cvae3.eval()
cvae4.eval()
cvae5.eval()
cvae6.eval()

X_train_z = np.load("../data_models_saved/data/X_train_z.npy")
X_train_z = torch.tensor(X_train_z, dtype=torch.float32).to(device)

X_valid_z = np.load("../data_models_saved/data/X_valid_z.npy")
X_valid_z = torch.tensor(X_valid_z, dtype=torch.float32).to(device)

X_test_z = np.load("../data_models_saved/data/X_test_z.npy")
X_test_z = torch.tensor(X_test_z, dtype=torch.float32).to(device)

X_all_z = torch.cat((X_train_z, X_valid_z, X_test_z), dim=0)


NumTrainSamples = X_train_z.shape[0]
NumValidSamples = X_valid_z.shape[0]

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

X_train_confounders = scaler.fit_transform(x_confounders[:NumTrainSamples, :])

### For the same metadata instance, generate num_samples number of samples with CAN-DO and cVAEs
### Compute the effective rank of the jacobian of the decoder
num_samples = 200
latent_dim = 44

eranks_mean = []
eranks_std = []

x_test_conf_instance = np.load('../data_models_saved/data/metadata_sampled_female.npy')[0]
x_test_conf_instance = np.tile(x_test_conf_instance.reshape(1, 4), (num_samples, 1))
x_test_conf_instance = scaler.transform(x_test_conf_instance)
x_test_conf_instance = torch.tensor(x_test_conf_instance, dtype=torch.float32, device=device)
### CAN-DO
z_synth_cando, _ = cando_cnf.reverse(x_test_conf_instance)
erank_cando_mean, erank_cando_std = jacobian_effective_rank(X=z_synth_cando, decoder=cando_decoder, c=None, ae=True)

eranks_mean.append(erank_cando_mean)
eranks_std.append(erank_cando_std)
print(f'ERank CAN-DO decoder: {erank_cando_mean} +- {erank_cando_std}')


### cVAE models
latent_dist = torch.distributions.MultivariateNormal(torch.zeros(latent_dim), torch.eye(latent_dim))
z_synth_normal = latent_dist.sample((num_samples,)).to('cuda')

cvae_models = [cvae1, cvae2, cvae3, cvae4, cvae5, cvae6]
for i, model in enumerate(cvae_models):
    erank_cvae_mean, erank_cvae_std = jacobian_effective_rank(X=z_synth_normal, decoder=model, c=x_test_conf_instance, ae=False)

    eranks_mean.append(erank_cvae_mean)
    eranks_std.append(erank_cvae_std)
    print(f'ERank cVAE beta= 10^-{i+1} : {erank_cvae_mean} +- {erank_cvae_std}')

np.save('../data_models_saved/data/eranks_all_mean.npy', np.array(eranks_mean))
np.save('../data_models_saved/data/eranks_all_std.npy', np.array(eranks_std))









