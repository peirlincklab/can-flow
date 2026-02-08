import torch
from torch.utils.data import DataLoader
import torch.utils.data as data
from models.AE import ConvAE, TrainerAE
import numpy as np


class Data(data.Dataset):

    def __init__(self, X):
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

batch_size_frac = 0.15

frac_train = 0.7
frac_valid = 0.15
epochs = 2000
lrate = 2e-4

NumSamples = 2274

latent_dimension = 44

NumTrainSamples = int(NumSamples * frac_train)
NumValidSamples = int(NumSamples * frac_valid)
NumTestSamples = NumSamples - NumTrainSamples - NumValidSamples


momenta = np.loadtxt("../data_models_saved/data/DeterministicAtlas__EstimatedParameters__Momenta.txt")

momenta = np.delete(momenta, 0, axis=0)
momenta = momenta.reshape((2274, 720, 3))

X_train_momenta = momenta[:NumTrainSamples, :, :].reshape((NumTrainSamples, 3, 8, 9, 10))
X_valid_momenta = momenta[NumTrainSamples:NumTrainSamples + NumValidSamples, :, :].reshape((NumValidSamples, 3, 8, 9, 10))
X_test_momenta = momenta[NumTrainSamples + NumValidSamples:, :, :].reshape((NumTestSamples, 3, 8, 9, 10))


dataset_train = Data(X=X_train_momenta)
dataset_valid = Data(X=X_valid_momenta)

train_loader = DataLoader(dataset=dataset_train, batch_size=int(batch_size_frac * NumTrainSamples), shuffle=True)
valid_loader = DataLoader(dataset=dataset_valid, batch_size=NumValidSamples, shuffle=False)


Convol_ae = ConvAE(latent_dim=latent_dimension).to(device)

optimizer = torch.optim.Adam(Convol_ae.parameters(), lr=lrate, weight_decay=1e-4)


trainer = TrainerAE(model=Convol_ae, optimizer=optimizer, epochs=epochs,
                         train_loader=train_loader, valid_loader=valid_loader,
                         latent_dim=latent_dimension)


Convol_ae = trainer.training()
trainer.plot_losses()

torch.save(Convol_ae, "../data_models_saved/models/ae_model.pth")
torch.save(Convol_ae.encoder, "../data_models_saved/models/ae_encoder.pth")
torch.save(Convol_ae.decoder, "../data_models_saved/models/ae_decoder.pth")

