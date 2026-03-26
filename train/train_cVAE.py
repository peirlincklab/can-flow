from torch.utils.data import DataLoader
import torch.utils.data as data
from models.cVAE import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os



class Data(data.Dataset):

    def __init__(self, X, y):

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


def set_seed(seed: int = 42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multiple GPUs

    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


scaler = MinMaxScaler()

batch_size_frac = 0.15
frac_train = 0.7
frac_valid = 0.15
epochs = 2000
lrate = 2e-4
beta = 1e-6

latent_dimension = 44

momenta = np.loadtxt("data_models_saved/data/DeterministicAtlas__EstimatedParameters__Momenta.txt")

momenta = np.delete(momenta, 0, axis=0)
momenta = momenta.reshape((2274, 720, 3))

### Exclude outliers and participants that withdrew from the study
momenta = np.delete(momenta, [1746, 1831], axis=0)

outliers = np.load('utils/outliers_indices.npy')
mask = np.ones(momenta.shape[0], dtype=bool)
mask[outliers] = False

momenta = momenta[mask]


NumSamples = momenta.shape[0]

NumTrainSamples = int(NumSamples * frac_train)
NumValidSamples = int(NumSamples * frac_valid)

X_train_momenta = momenta[:NumTrainSamples, :, :].reshape((NumTrainSamples, 3, 8, 9, 10))
X_valid_momenta = momenta[NumTrainSamples:NumTrainSamples + NumValidSamples, :, :].reshape((NumValidSamples, 3, 8, 9, 10))



x_confounders = pd.read_excel("/home/kostas/home/metadata_final.xlsx")
x_confounders.drop(['Participant ID', 'Height', 'Weight', 'Diastolic BP',
                    'Systolic BP', 'Unnamed: 8', 'Unnamed: 9', 'subject_id'], axis=1, inplace=True)
x_confounders = pd.get_dummies(x_confounders, columns=['Sex'])
x_confounders['Sex_Female'] = x_confounders['Sex_Female'].replace({True: 1, False: 0})
x_confounders['Sex_Male'] = x_confounders['Sex_Male'].replace({True: 1, False: 0})
x_confounders = x_confounders[['BMI', 'Age', 'Sex_Female', 'Sex_Male']]
x_confounders = x_confounders.to_numpy()


### Delete outliers and participants that withdrew from the study
x_confounders = np.delete(x_confounders, [1746, 1831], axis=0)
x_confounders = x_confounders[mask]


X_train_confounders = scaler.fit_transform(x_confounders[:NumTrainSamples, :])
X_valid_confounders = scaler.transform(x_confounders[NumTrainSamples:NumTrainSamples+NumValidSamples, :])

X_train_confounders = torch.tensor(X_train_confounders, dtype=torch.float32).to(device)
X_valid_confounders = torch.tensor(X_valid_confounders, dtype=torch.float32).to(device)


dataset_train = Data(X=X_train_momenta, y=X_train_confounders)
dataset_valid = Data(X=X_valid_momenta, y=X_valid_confounders)

train_loader = DataLoader(dataset=dataset_train, batch_size=int(frac_train * NumTrainSamples), shuffle=True)
valid_loader = DataLoader(dataset=dataset_valid, batch_size=NumValidSamples, shuffle=False)

cvae = cVAE(latent_dim=latent_dimension, cond_dim=4, embed_dim_conf=12)

optimizer = torch.optim.Adam(cvae.parameters(), lr=lrate, weight_decay=1e-5)

trainer = Trainer_cVAE(model=cvae, optimizer=optimizer, epochs=epochs, train_loader=train_loader,
                       valid_loader=valid_loader, latent_dim=latent_dimension, beta=beta)

cvae_trained = trainer.training()
trainer.plot_losses()

torch.save(cvae_trained, f"data_models_saved/models/cvae_model_beta_{beta}.pth")
torch.save(cvae_trained.encoder, f"data_models_saved/models/cvae_encoder_beta_{beta}.pth")
torch.save(cvae_trained.decoder, f"data_models_saved/models/cvae_decoder_beta_{beta}.pth")
