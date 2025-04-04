import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.manifold import TSNE

def load_cp_momenta(file_name):

    # Get the path to the Downloads folder dynamically
    downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")

    file_path = os.path.join(downloads_folder, file_name)
    data = np.loadtxt(file_path)

    return data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

num_dense = 2730
frac_train = 0.5
frac_valid = 0.45

NumTrainSamples = int(456 * frac_train)
NumValidSamples = int(456 * frac_valid)
NumTestSamples = 456 - NumTrainSamples - NumValidSamples

file_path_confounders = r'C:\Users\kkevopoulos\Downloads\cardiac_function_mesh_complete_final.xlsx'
file_name_momenta = r"cardiac_atlas_kostas\output\DeterministicAtlas__EstimatedParameters__Momenta.txt"

x_confounders = pd.read_excel(file_path_confounders)
x_confounders.drop(['BSA', 'Year of birth', 'Height', 'Weight', 'Diastolic BP mean reading',
                 'Systolic BP mean reading', 'mesh_file', 'MAP'], axis=1, inplace=True)
x_confounders['Sex'] = x_confounders['Sex'].replace({'Male': 0, 'Female': 1})
x_confounders = x_confounders.to_numpy()

x_confounders = x_confounders[:, 1:]

sex_info = x_confounders[:NumTrainSamples, 0]
male_indices = np.where(sex_info.flatten() == 0)[0]
female_indices = np.where(sex_info.flatten() == 1)[0]



momenta = load_cp_momenta(file_name_momenta)

momenta = np.delete(momenta, 0, axis=0)
momenta = momenta.reshape((456, 2730, 3))

X_train_momenta = momenta[:NumTrainSamples, :, :].reshape((NumTrainSamples, 3, 13, 14, 15))
X_valid_momenta = momenta[NumTrainSamples:NumTrainSamples + NumValidSamples, :, :].reshape((NumValidSamples, 3, 13, 14, 15))
X_test_momenta = momenta[NumTrainSamples + NumValidSamples:, :, :].reshape((NumTestSamples, 3, 13, 14, 15))

X_train_momenta = torch.tensor(X_train_momenta, dtype=torch.float32).to("cuda")
X_valid_momenta = torch.tensor(X_valid_momenta, dtype=torch.float32).to("cuda")
X_test_momenta = torch.tensor(X_test_momenta, dtype=torch.float32).to("cuda")


encoder = torch.load("../data_models_saved/encoder.pth")
encoder.eval()


X_train_z = encoder(X_train_momenta)
X_train_z = X_train_z.detach().cpu().numpy()
np.save("../data_models_saved/X_train_z.npy", X_train_z)

X_valid_z = encoder(X_valid_momenta)
X_valid_z = X_valid_z.detach().cpu().numpy()
np.save("../data_models_saved/X_valid_z.npy", X_valid_z)

X_test_z = encoder(X_test_momenta)
X_test_z = X_test_z.detach().cpu().numpy()
np.save("../data_models_saved/X_test_z.npy", X_test_z)


## Experiment1: Visualize the 100-dim latent space with t-sne
z_latent_train = encoder(X_train_momenta)
z_latent_train = z_latent_train.detach().cpu().numpy()

z_tsne = TSNE(n_components=2, learning_rate='auto',
                                    init='random', perplexity=20).fit_transform(z_latent_train)

plt.figure()
plt.scatter(z_tsne[female_indices, 0], z_tsne[female_indices, 1], color='red', label='female')
plt.scatter(z_tsne[male_indices, 0], z_tsne[male_indices, 1], color='blue', label='male')
plt.legend()
plt.title('t-SNE visualization of reduced bases')
plt.show()



### Experiment 2: For all shapes in the validation dataset, plot the corresponding latent represenation in a 2d plot
z_latent_valid = encoder(X_valid_momenta)
z_latent_valid = z_latent_valid.detach().cpu().numpy()


for i in range(z_latent_valid.shape[0]):

    plt.imshow(z_latent_valid[i].reshape(5, 10), cmap='bwr')
    plt.colorbar()
    plt.title(f"sample {i}")
    plt.show()

