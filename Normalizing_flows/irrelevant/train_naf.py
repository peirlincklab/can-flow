import os

from torch.utils.data import DataLoader
import torch.utils.data as data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from NAF_Test import *

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

def load_cp_momenta(file_name):

    # Get the path to the Downloads folder dynamically
    downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")

    file_path = os.path.join(downloads_folder, file_name)
    data = np.loadtxt(file_path)

    return data

scaler = MinMaxScaler()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


file_path_confounders = r'C:\Users\kkevopoulos\Downloads\cardiac_function_mesh_complete_final.xlsx'
x_confounders = pd.read_excel(file_path_confounders)
x_confounders.drop(['BSA', 'Year of birth', 'Height', 'Weight', 'Diastolic BP mean reading',
                 'Systolic BP mean reading', 'mesh_file', 'MAP'], axis=1, inplace=True)
x_confounders['Sex'] = x_confounders['Sex'].map({'Female':1, 'Male':0})
x_confounders = x_confounders.to_numpy()
x_confounders = x_confounders[:, 1:]


batch_size_frac = 0.35
epochs = 300
lrate = 7e-4


X_train_z = np.load("../data_models_saved/X_train_z.npy")
X_train_z = torch.tensor(X_train_z, dtype=torch.float32).to(device)

X_valid_z = np.load("../data_models_saved/X_valid_z.npy")
X_valid_z = torch.tensor(X_valid_z, dtype=torch.float32).to(device)

X_test_z = np.load("../data_models_saved/X_test_z.npy")
X_test_z = torch.tensor(X_test_z, dtype=torch.float32).to(device)

NumTrainSamples = X_train_z.shape[0]
NumValidSamples = X_valid_z.shape[0]
NumTestSamples = X_test_z.shape[0]


X_train_confounders = scaler.fit_transform(x_confounders[:NumTrainSamples, :])
X_valid_confounders = scaler.fit_transform(x_confounders[NumTrainSamples:NumTrainSamples+NumValidSamples, :])
X_test_confounders = scaler.fit_transform(x_confounders[NumTrainSamples+NumValidSamples:, :])

X_train_confounders = torch.tensor(X_train_confounders, dtype=torch.float32).to(device)
X_valid_confounders = torch.tensor(X_valid_confounders, dtype=torch.float32).to(device)
X_test_confounders = torch.tensor(X_test_confounders, dtype=torch.float32).to(device)


X_train = torch.cat((X_train_z, X_train_confounders), dim=1)
X_valid = torch.cat((X_valid_z, X_valid_confounders), dim=1)
X_test = torch.cat((X_test_z, X_test_confounders), dim=1)

n_dim = X_train.shape[1]

dataset_train = Data(X=X_train)
dataset_valid = Data(X=X_valid)

train_loader = DataLoader(dataset=dataset_train, batch_size=int(batch_size_frac * NumTrainSamples), shuffle=True)
valid_loader = DataLoader(dataset=dataset_valid, batch_size=int(batch_size_frac * NumValidSamples), shuffle=False)

naf_model = NormalizingFlow(in_dim=n_dim, out_dim=n_dim, n_flow=90)

optimizer = torch.optim.Adam(naf_model.parameters(), lr=lrate, weight_decay=1e-4)

trainer = Trainer(model=naf_model, optimizer=optimizer, epochs=epochs, train_loader=train_loader, valid_loader=valid_loader,
                  latent_dim=n_dim, l1_w=0, omega_tol=0.015, alpha=1, rho_dual=1)


norm_flow = trainer.training()
trainer.plot_losses()
print(f"min {norm_flow.mask.min()}, max {norm_flow.mask.max()}")


def threshold_W(W):
    dim = W.shape[0]

    for i in range(dim):
        for j in range(dim):

            if np.abs(W[i, j]) < 0.002:
                W[i, j] = 0

    return W

mask = threshold_W(norm_flow.mask.detach().cpu().numpy())

plt.imshow(norm_flow.mask.detach().cpu().numpy())
plt.colorbar()
plt.show()
torch.save(norm_flow, "flow_model_naf.pth")