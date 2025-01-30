import numpy as np
import torch
from Main_Encoder3DConv import *
from torch.utils.data import DataLoader
import torch.utils.data as data
from sklearn.neighbors import NearestNeighbors
import os
import pandas as pd
import warnings
from sklearn.preprocessing import minmax_scale

warnings.filterwarnings('ignore')


########################################################################################################################
###### Functions that will be needed during the run ######
########################################################################################################################
def load_cp_momenta(file_name):

    # Get the path to the Downloads folder dynamically
    downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")

    file_path = os.path.join(downloads_folder, file_name)
    data = np.loadtxt(file_path)

    return data


class Data(data.Dataset):

    def __init__(self, X, y):
        """
        Class that preprocesses the data that go into the FNN architecture.
        This class is needed for the Dataloader.
        :param X: Inputs of the neural network (e.g. the parameters "mu")
        :param y: Outputs of the neural network (e.g. x or f(x) for a specific timestep t*)
        """

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        # Number of data points we have.
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        data_point_x = self.X[idx]
        data_point_y = self.y[idx]
        return data_point_x, data_point_y


########################################################################################################################
###### Configurations of the run ######
########################################################################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

batch_size_frac = 0.05
num_dense = 2730
frac_train = 0.9
frac_valid = 0.05
epochs = 300 # 1000
lrate = 4e-4

latent_dimension = 16

NumTrainSamples = int(456 * frac_train)
NumValidSamples = int(456 * frac_valid)
NumTestSamples = 456 - NumTrainSamples - NumValidSamples


########################################################################################################################
###### Data loading & pre processing ######
########################################################################################################################


file_name_cp = "cardiac_atlas_kostas\output\DeterministicAtlas__EstimatedParameters__ControlPoints.txt"
file_name_momenta = "cardiac_atlas_kostas\output\DeterministicAtlas__EstimatedParameters__Momenta.txt"
file_path_confounders = r'C:\Users\kkevopoulos\Downloads\cardiac_function_mesh_complete_final.xlsx'


control_points = load_cp_momenta(file_name_cp)
momenta = load_cp_momenta(file_name_momenta)

x_confounders = pd.read_excel(file_path_confounders)
x_confounders.drop(['BSA', 'Year of birth', 'Height', 'Weight', 'Diastolic BP mean reading',
                 'Systolic BP mean reading', 'mesh_file', 'MAP'], axis=1, inplace=True)
x_confounders['Sex'] = x_confounders['Sex'].replace({'Male': 0, 'Female': 1})
x_confounders = x_confounders.to_numpy()

x_confounders = x_confounders[:, 1:]


momenta = np.delete(momenta, 0, axis=0)
momenta = momenta.reshape((456, 2730, 3))

density_feature = np.zeros((456, 2730, 1))



for i in range(momenta.shape[0]):
    nbrs = NearestNeighbors(n_neighbors=8).fit(momenta[i])
    distances, _ = nbrs.kneighbors(momenta[i])
    density = 1 / distances[:, -1]
    density_normalized = (density - density.min()) / (density.max() - density.min())

    density_feature[i] = density_normalized.reshape(-1, 1)

momenta = np.concatenate((momenta, density_feature), axis=-1)

X_train_momenta = momenta[:NumTrainSamples, :, :].reshape((NumTrainSamples, 4, 13, 14, 15))
X_valid_momenta = momenta[NumTrainSamples:NumTrainSamples + NumValidSamples, :, :].reshape((NumValidSamples, 4, 13, 14, 15))
X_test_momenta = momenta[NumTrainSamples + NumValidSamples:, :, :].reshape((NumTestSamples, 4, 13, 14, 15))


X_train_confounders = minmax_scale(x_confounders[:NumTrainSamples, :], axis=0)
X_valid_confounders = minmax_scale(x_confounders[NumTrainSamples:NumTrainSamples+NumValidSamples, :], axis=0)
X_test_confounders = minmax_scale(x_confounders[NumTrainSamples+NumValidSamples:, :], axis=0)

########################################################################################################################
###### Build the VAE model ######
########################################################################################################################


dataset_train = Data(X=X_train_momenta, y=X_train_confounders)
dataset_valid = Data(X=X_valid_momenta, y=X_valid_confounders)

train_loader = DataLoader(dataset=dataset_train, batch_size=5, shuffle=True)
valid_loader = DataLoader(dataset=dataset_valid, batch_size=int(batch_size_frac * NumValidSamples), shuffle=False)

Convol_vae = ConvVAE(latent_dim=latent_dimension).to(device)

optimizer = torch.optim.Adam(Convol_vae.parameters(), lr=lrate)


########################################################################################################################
###### Train the model ######
########################################################################################################################

trainer = TrainerConvVAE(model=Convol_vae, optimizer=optimizer, epochs=epochs,
                         train_loader=train_loader, valid_loader=valid_loader, latent_dim=latent_dimension)

Convol_vae = trainer.training()
trainer.plot_losses()

torch.save(Convol_vae, "model")
torch.save(Convol_vae.encoder, "encoder.pth")
torch.save(Convol_vae.decoder, "decoder.pth")

