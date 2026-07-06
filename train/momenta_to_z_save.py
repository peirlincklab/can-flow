import numpy as np
import torch



### This file encodes real momenta into the latent space using the CAN-FLOW autoencoder, and saves the latent representations
device = 'cuda'

momenta = np.loadtxt("data_models_saved/data/DeterministicAtlas__EstimatedParameters__Momenta.txt")

momenta = np.delete(momenta, 0, axis=0)
momenta = momenta.reshape((2274, 720, 3))

frac_train = 0.7
frac_valid = 0.15

### Exclude outliers and participants that withdrew from the study
momenta = np.delete(momenta, [1746, 1831], axis=0)

outliers = np.load('utils/outliers_indices.npy')
mask = np.ones(momenta.shape[0], dtype=bool)
mask[outliers] = False

momenta = momenta[mask]


NumSamples = momenta.shape[0]

NumTrainSamples = int(NumSamples * frac_train)
NumValidSamples = int(NumSamples * frac_valid)
NumTestSamples = NumSamples - NumTrainSamples - NumValidSamples


X_train_momenta = momenta[:NumTrainSamples, :, :].reshape((NumTrainSamples, 3, 8, 9, 10))
X_valid_momenta = momenta[NumTrainSamples:NumTrainSamples + NumValidSamples, :, :].reshape((NumValidSamples, 3, 8, 9, 10))
X_test_momenta = momenta[NumTrainSamples + NumValidSamples:, :, :].reshape((NumTestSamples, 3, 8, 9, 10))

X_train_momenta = torch.tensor(X_train_momenta, dtype=torch.float32).to(device)
X_valid_momenta = torch.tensor(X_valid_momenta, dtype=torch.float32).to(device)
X_test_momenta = torch.tensor(X_test_momenta, dtype=torch.float32).to(device)


conv_encoder = torch.load("data_models_saved/models/ae_encoder.pth", weights_only=False)
conv_encoder.eval()

z_latent_train = conv_encoder(X_train_momenta)
z_latent_valid = conv_encoder(X_valid_momenta)
z_latent_test = conv_encoder(X_test_momenta)

z_latent_train = z_latent_train.detach().cpu().numpy()
z_latent_valid = z_latent_valid.detach().cpu().numpy()
z_latent_test = z_latent_test.detach().cpu().numpy()

np.save('data_models_saved/data/X_train_z.npy', z_latent_train)
np.save('data_models_saved/data/X_valid_z.npy', z_latent_valid)
np.save('data_models_saved/data/X_test_z.npy', z_latent_test)
