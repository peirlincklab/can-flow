### Optimize some hyperparameters of the normalizing flow
### The criterion will be the WD between the AE latent space and the autoencoder latent space
### We evaluate this criterion for the validation and test set

import optuna
import torch
from models.cNF import *
from torch.utils.data import DataLoader
import torch.utils.data as data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import optuna.visualization as vis
import ot



def Wasserstein_distance(X, Y):
    "Entropically regularized optimal transport cost between two empirical distributions"
    M = ot.dist(X, Y, metric='euclidean')
    a = np.ones(len(X)) / len(X)
    b = np.ones(len(Y)) / len(Y)

    W = ot.sinkhorn2(a, b, M, reg=2.1)

    return W


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

def get_activation(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.2)
    elif name == 'elu':
        return nn.ELU()
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'silu':
        return nn.SiLU()
    else:
        raise ValueError(f"Unknown activation: {name}")



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

scaler = MinMaxScaler()


#x_confounders = pd.read_excel(r"/home/kevopou1/metadata_final.xlsx")
x_confounders = pd.read_excel(r"/home/kostas/home/metadata_final.xlsx")
x_confounders.drop(['Participant ID', 'Height', 'Weight', 'Diastolic BP',
                    'Systolic BP', 'Unnamed: 8', 'Unnamed: 9', 'subject_id'], axis=1, inplace=True)
x_confounders = pd.get_dummies(x_confounders, columns=['Sex'])
x_confounders['Sex_Female'] = x_confounders['Sex_Female'].replace({True: 1, False: 0})
x_confounders['Sex_Male'] = x_confounders['Sex_Male'].replace({True: 1, False: 0})
x_confounders = x_confounders[['BMI', 'Age', 'Sex_Female', 'Sex_Male']]
x_confounders = x_confounders.to_numpy()


batch_size_frac = 0.15
frac_train = 0.7
frac_valid = 0.15
epochs = 1200


X_train_z = np.load("data_models_saved/data/X_train_z.npy")
X_train_z = torch.tensor(X_train_z, dtype=torch.float32).to(device)

X_valid_z = np.load("data_models_saved/data/X_valid_z.npy")
X_valid_z = torch.tensor(X_valid_z, dtype=torch.float32).to(device)

X_test_z = np.load("data_models_saved/data/X_test_z.npy")
X_test_z = torch.tensor(X_test_z, dtype=torch.float32).to(device)

NumTrainSamples = X_train_z.shape[0]
NumValidSamples = X_valid_z.shape[0]

X_train_confounders = scaler.fit_transform(x_confounders[:NumTrainSamples, :])
X_valid_confounders = scaler.transform(x_confounders[NumTrainSamples:NumTrainSamples+NumValidSamples, :])
X_test_confounders = scaler.transform(x_confounders[NumTrainSamples+NumValidSamples:])

X_train_confounders = torch.tensor(X_train_confounders, dtype=torch.float32).to(device)
X_valid_confounders = torch.tensor(X_valid_confounders, dtype=torch.float32).to(device)
X_test_confounders = torch.tensor(X_test_confounders, dtype=torch.float32).to(device)


dim_latent = X_train_z.shape[1]

dataset_train = Data(X=X_train_z, y=X_train_confounders)
dataset_valid = Data(X=X_valid_z, y=X_valid_confounders)

train_loader = DataLoader(dataset=dataset_train, batch_size=int(batch_size_frac * NumTrainSamples), shuffle=True)
valid_loader = DataLoader(dataset=dataset_valid, batch_size=NumValidSamples, shuffle=False)


def objective(trial):

    n_flow_layers = trial.suggest_int('n_flow_layers', 5, 40)
    activ_suggest = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu', 'gelu', 'tanh', 'silu'])
    lrate = trial.suggest_float("lr", 1e-5, 5e-4, log=True)

    use_weight_decay = trial.suggest_categorical("use_weight_decay", [True, False])

    if use_weight_decay:
        weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-4, log=True)
    else:
        weight_decay = 0.0

    activation = get_activation(name=activ_suggest)

    norm_flow = FlowHeart(in_dim=dim_latent, out_dim=dim_latent, n_flow=n_flow_layers, in_dim_conf=4, out_dim_conf=12,
                          device=device, activations=activation)

    optimizer = torch.optim.Adam(norm_flow.parameters(), lr=lrate, weight_decay=weight_decay)

    trainer = TrainerNF(model=norm_flow, optimizer=optimizer, epochs=epochs,
                        train_loader=train_loader, valid_loader=valid_loader,
                        latent_dim=dim_latent)

    norm_flow = trainer.training()

    with torch.no_grad():
        norm_flow.eval()

        z_approx_valid, _ = norm_flow.reverse(X_valid_confounders)
        z_approx_test, _ = norm_flow.reverse(X_test_confounders)

        z_approx_valid_test = torch.cat((z_approx_valid, z_approx_test), dim=0)
        z_approx_valid_test = z_approx_valid_test.detach().cpu().numpy()

        z_ae_valid_test = torch.cat(X_valid_z, X_test_z)
        z_ae_valid_test = z_ae_valid_test.detach().cpu().numpy()

        wd = Wasserstein_distance(z_ae_valid_test, z_approx_valid_test)

        return wd

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=3)

fig1 = vis.plot_optimization_history(study)
fig2 = vis.plot_param_importances(study)
fig3 = vis.plot_slice(study)
fig4 = vis.plot_parallel_coordinate(study)

fig1.show()
fig2.show()
fig3.show()
fig4.show()

print("Best accuracy:", study.best_value)
print("Best hyperparameters:", study.best_params)



