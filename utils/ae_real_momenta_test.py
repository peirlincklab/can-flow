import numpy as np
import torch
from utils import save_write_momenta

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


ae_decoder = torch.load("../data_models_saved/models/ae_decoder.pth", weights_only=False)
ae_decoder.eval()

X_train_z = np.load("../data_models_saved/data/X_train_z.npy")
X_train_z = torch.tensor(X_train_z, dtype=torch.float32).to(device)

X_valid_z = np.load("../data_models_saved/data/X_valid_z.npy")
X_valid_z = torch.tensor(X_valid_z, dtype=torch.float32).to(device)

X_test_z = np.load("../data_models_saved/data/X_test_z.npy")
X_test_z = torch.tensor(X_test_z, dtype=torch.float32).to(device)

X_all_z = torch.cat((X_train_z, X_valid_z, X_test_z), dim=0)


reconstructed_momenta = ae_decoder(X_all_z)
reconstructed_momenta = reconstructed_momenta.reshape(X_all_z.shape[0], 720, 3)
reconstructed_momenta = reconstructed_momenta.detach().cpu().numpy()

save_write_momenta.save_momenta(type_momenta="Reference_Reconstructed", momenta_tosave=reconstructed_momenta)