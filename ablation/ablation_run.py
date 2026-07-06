import torch
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import torch.utils.data as data
from models.cNF import *
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


os.makedirs('ablation/models', exist_ok=True)

### Set up some common parts of the training for all configurations
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

### Delete outliers and participants that withdrew from the study
x_confounders = np.delete(x_confounders, [1746, 1831], axis=0)

outliers = np.load('utils/outliers_indices.npy')
mask = np.ones(x_confounders.shape[0], dtype=bool)
mask[outliers] = False

x_confounders = x_confounders[mask]



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

batch_size_frac = 0.15
frac_train = 0.7
frac_valid = 0.15
epochs = 1200
lrate = 2.5e-4

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

X_train_confounders = torch.tensor(X_train_confounders, dtype=torch.float32).to(device)
X_valid_confounders = torch.tensor(X_valid_confounders, dtype=torch.float32).to(device)

dataset_train = Data(X=X_train_z, y=X_train_confounders)
dataset_valid = Data(X=X_valid_z, y=X_valid_confounders)

train_loader = DataLoader(dataset=dataset_train, batch_size=int(batch_size_frac * NumTrainSamples), shuffle=True)
valid_loader = DataLoader(dataset=dataset_valid, batch_size=NumValidSamples, shuffle=False)

dim_latent = X_train_z.shape[1]
print(f"Latent dimension is: {dim_latent}")


### Define some useful dicts
activation_dict = {
    "relu": nn.ReLU(),
    "elu": nn.ELU(),
    "leaky_relu": nn.LeakyReLU(),
    "silu": nn.SiLU(),
    "gelu": nn.GELU()
}



base_config = {
    "n_flow": 15,
    "out_dim_conf": 12,
    "activation": "gelu"
}



### different things to ablate
n_flow = [5, 10, 20, 25, 30]
out_dim_conf =[6, 9, 15, 18, 21]
activations = ["relu", "elu", "leaky_relu", "silu"]

ablate_params = [n_flow, out_dim_conf, activations]
ablate_params_str = ["n_flow", "out_dim_conf", "activation"]

num_runs_train = 3

for ablate, ablate_str in zip(ablate_params, ablate_params_str):

    for i in ablate:

        config_name = f"{ablate_str}_{i}"
        with open(f"ablation/configs/{config_name}.json", "r") as f:
            config = json.load(f)

        n_flow_config = config["n_flow"]
        out_dim_conf_config = config["out_dim_conf"]
        activation_config = activation_dict[config["activation"]]

        for num_run in range(num_runs_train):
            ### Train the model for each configuration
            norm_flow = FlowHeart(in_dim=dim_latent, out_dim=dim_latent, n_flow=n_flow_config, in_dim_conf=4,
                                  out_dim_conf=out_dim_conf_config, device=device, activations=activation_config)

            optimizer = torch.optim.Adam(norm_flow.parameters(), lr=lrate, weight_decay=1e-5)

            trainer = TrainerNF(model=norm_flow, optimizer=optimizer, epochs=epochs,
                                train_loader=train_loader, valid_loader=valid_loader,
                                latent_dim=dim_latent)

            norm_flow = trainer.training()

            ### After the normalizing flow model is trained, save the model
            torch.save(norm_flow, f"ablation/models/cnf_model_ablation_{ablate_str}_{i}_{num_run}.pth")
            print(f"Finished run: {ablate_str},  {i}, num_run:{num_run}")


