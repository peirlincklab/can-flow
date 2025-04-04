from torch.utils.data import DataLoader
import torch.utils.data as data
from models.ConvAE import *
import os



def load_cp_momenta(file_name):

    # Get the path to the Downloads folder dynamically
    downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")

    file_path = os.path.join(downloads_folder, file_name)
    data = np.loadtxt(file_path)

    return data


class Data(data.Dataset):

    def __init__(self, X):
        """
        Class that preprocesses the data that go into the FNN architecture.
        This class is needed for the Dataloader.
        :param X: Inputs of the neural network (e.g. the parameters "mu")
        :param y: Outputs of the neural network (e.g. x or f(x) for a specific timestep t*)
        """

        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        # Number of data points we have.
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        data_point_x = self.X[idx]

        return data_point_x




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

batch_size_frac = 0.4
num_dense = 2730
frac_train = 0.5
frac_valid = 0.45
epochs = 1500
lrate = 3e-4


latent_dimension = 50

NumTrainSamples = int(456 * frac_train)
NumValidSamples = int(456 * frac_valid)
NumTestSamples = 456 - NumTrainSamples - NumValidSamples



file_name_momenta = r"cardiac_atlas_kostas\output\DeterministicAtlas__EstimatedParameters__Momenta.txt"
momenta = load_cp_momenta(file_name_momenta)

momenta = np.delete(momenta, 0, axis=0)
momenta = momenta.reshape((456, 2730, 3))

X_train_momenta = momenta[:NumTrainSamples, :, :].reshape((NumTrainSamples, 3, 13, 14, 15))
X_valid_momenta = momenta[NumTrainSamples:NumTrainSamples + NumValidSamples, :, :].reshape((NumValidSamples, 3, 13, 14, 15))
X_test_momenta = momenta[NumTrainSamples + NumValidSamples:, :, :].reshape((NumTestSamples, 3, 13, 14, 15))


dataset_train = Data(X=X_train_momenta)
dataset_valid = Data(X=X_valid_momenta)

train_loader = DataLoader(dataset=dataset_train, batch_size=5, shuffle=True)
valid_loader = DataLoader(dataset=dataset_valid, batch_size=int(batch_size_frac * NumValidSamples), shuffle=False)


Convol_ae = ConvAE(latent_dim=latent_dimension).to(device)

optimizer = torch.optim.Adam(Convol_ae.parameters(), lr=lrate, weight_decay=1e-4)


trainer = TrainerConvAE(model=Convol_ae, optimizer=optimizer, epochs=epochs,
                         train_loader=train_loader, valid_loader=valid_loader,
                         latent_dim=latent_dimension)


Convol_ae = trainer.training()
trainer.plot_losses()

torch.save(Convol_ae, "../data_models_saved/model.pth")
torch.save(Convol_ae.encoder, "../data_models_saved/encoder.pth")
torch.save(Convol_ae.decoder, "../data_models_saved/decoder.pth")

