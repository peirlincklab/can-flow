import os
from models.cNormalizingFlow import *
from torch.utils.data import DataLoader
import torch.utils.data as data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



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


def load_cp_momenta(file_name):

    # Get the path to the Downloads folder dynamically
    downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")

    file_path = os.path.join(downloads_folder, file_name)
    data = np.loadtxt(file_path)

    return data

scaler = MinMaxScaler()

file_path_confounders = r'C:\Users\kkevopoulos\Downloads\cardiac_function_mesh_complete_final.xlsx'
x_confounders = pd.read_excel(file_path_confounders)
x_confounders.drop(['BSA', 'Year of birth', 'Height', 'Weight', 'Diastolic BP mean reading',
                 'Systolic BP mean reading', 'mesh_file', 'MAP'], axis=1, inplace=True)
x_confounders = pd.get_dummies(x_confounders, columns=['Sex'])
x_confounders['Sex_Female'] = x_confounders['Sex_Female'].replace({True: 1, False: 0})
x_confounders['Sex_Male'] = x_confounders['Sex_Male'].replace({True: 1, False: 0})
x_confounders = x_confounders.to_numpy()
x_confounders = x_confounders[:, 1:]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

batch_size_frac = 0.35
num_dense = 2730
frac_train = 0.5
frac_valid = 0.45
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


dim = X_train_z.shape[1]


dataset_train = Data(X=X_train_z, y=X_train_confounders)
dataset_valid = Data(X=X_valid_z, y=X_valid_confounders)

train_loader = DataLoader(dataset=dataset_train, batch_size=int(batch_size_frac * NumTrainSamples), shuffle=True)
valid_loader = DataLoader(dataset=dataset_valid, batch_size=int(batch_size_frac * NumValidSamples), shuffle=False)

norm_flow = FlowHeart(in_dim=dim, out_dim=50, n_flow=90, in_dim_conf=4, out_dim_conf=20)

optimizer = torch.optim.Adam(norm_flow.parameters(), lr=lrate, weight_decay=1e-4)


trainer = TrainerNF(model=norm_flow, optimizer=optimizer, epochs=epochs,
                         train_loader=train_loader, valid_loader=valid_loader, latent_dim=dim)


norm_flow = trainer.training()
trainer.plot_losses()


### After the normalizing flow model is trained, save the model
# synthetic_z = norm_flow.reverse(x_conf=X_train_confounders)
# synthetic_z = synthetic_z.detach().cpu().numpy()
# np.save("synthetic_z.npy", synthetic_z)
torch.save(norm_flow, "../data_models_saved/flow_model.pth")








