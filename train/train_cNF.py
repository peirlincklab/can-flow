from models.cNF import *
from torch.utils.data import DataLoader
import torch.utils.data as data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

class Data(data.Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        ### Number of data points we have.
        return self.X.shape[0]

    def __getitem__(self, idx):
        ### Return the idx-th data point of the dataset
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


scaler = MinMaxScaler()

# x_confounders = pd.read_excel(r"/home/kevopou1/metadata_final.xlsx")
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

dim_latent = 50

X_train_z = np.load(f"ablation/ablation_latent_dimensionality/models_saved/X_train_z_{dim_latent}.npy")
X_train_z = torch.tensor(X_train_z, dtype=torch.float32).to(device)

X_valid_z = np.load(f"ablation/ablation_latent_dimensionality/models_saved/X_valid_z_{dim_latent}.npy")
X_valid_z = torch.tensor(X_valid_z, dtype=torch.float32).to(device)

X_test_z = np.load(f"ablation/ablation_latent_dimensionality/models_saved/X_test_z_{dim_latent}.npy")
X_test_z = torch.tensor(X_test_z, dtype=torch.float32).to(device)

NumTrainSamples = X_train_z.shape[0]
NumValidSamples = X_valid_z.shape[0]

X_train_confounders = scaler.fit_transform(x_confounders[:NumTrainSamples, :])
X_valid_confounders = scaler.transform(x_confounders[NumTrainSamples:NumTrainSamples+NumValidSamples, :])

X_train_confounders = torch.tensor(X_train_confounders, dtype=torch.float32).to(device)
X_valid_confounders = torch.tensor(X_valid_confounders, dtype=torch.float32).to(device)


#dim_latent = X_train_z.shape[1]
#print(f"Latent dimension is: {dim_latent}")


dataset_train = Data(X=X_train_z, y=X_train_confounders)
dataset_valid = Data(X=X_valid_z, y=X_valid_confounders)

train_loader = DataLoader(dataset=dataset_train, batch_size=int(batch_size_frac * NumTrainSamples), shuffle=True)
valid_loader = DataLoader(dataset=dataset_valid, batch_size=NumValidSamples, shuffle=False)

norm_flow = FlowHeart(in_dim=dim_latent, out_dim=dim_latent, n_flow=15, in_dim_conf=4, out_dim_conf=12, device=device, activations=nn.GELU())

optimizer = torch.optim.Adam(norm_flow.parameters(), lr=lrate, weight_decay=1e-5)


trainer = TrainerNF(model=norm_flow, optimizer=optimizer, epochs=epochs, 
                    train_loader=train_loader, valid_loader=valid_loader, 
                    latent_dim=dim_latent)


norm_flow = trainer.training()
trainer.plot_losses()


### After the normalizing flow model is trained, save the model
torch.save(norm_flow, f"ablation/ablation_latent_dimensionality/models_saved/cnf_model_latent_{dim_latent}.pth")
