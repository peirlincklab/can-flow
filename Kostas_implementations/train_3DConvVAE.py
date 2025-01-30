import numpy as np
import torch
from Main_Encoder3DConv import *
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.utils.data as data
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from sklearn.manifold import TSNE
import os
import pandas as pd
import warnings
from sklearn.preprocessing import minmax_scale

warnings.filterwarnings('ignore')


########################################################################################################################
###### Functions that will be needed during the run ######
########################################################################################################################

def savetxt(object, type, i, num_test=None):
    """
    Saves a NumPy array to a text file in the 'momenta/generated' folder on the Desktop.

    Parameters:
    - array (np.ndarray): The NumPy array to save.
    - filename (str): The name of the text file (default is "array.txt").
    """
    # Define the path to the 'momenta/generated' folder on the Desktop
    desktop_path = os.path.expanduser(f"~/Desktop/momenta/{type}")

    # Create the folder if it doesn't exist
    os.makedirs(desktop_path, exist_ok=True)

    if type == "generated":
        filename = f"{type}_{i}"
    else:
        filename = f"{type}_{i + (456-num_test)}"

    # Full path to the output file
    file_path = os.path.join(desktop_path, filename)

    # Save the array to the file
    np.savetxt(file_path, object, delimiter='\t')

    print(f"Array saved to {file_path}")


def load_cp_momenta(file_name):

    # Get the path to the Downloads folder dynamically
    downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")

    file_path = os.path.join(downloads_folder, file_name)
    data = np.loadtxt(file_path)

    return data


def debug_visuals(dist):
    z_latent = dist.mean.detach().cpu().numpy()

    z_embedded = TSNE(n_components=2, learning_rate='auto',
                      init='random', perplexity=3).fit_transform(z_latent)

    plt.scatter(z_embedded[:, 0], z_embedded[:, 1])
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.title("t-SNE visualization of a 2D latent space")
    plt.show()

    for i in range(latent_dimension):
        plt.figure()
        plt.hist(z_latent[:, i], bins=25, color='b', alpha=0.7)
        plt.title(fr"z_{i}")
        plt.show()


def visualization(num_neighbors, x_pred, x_ref, title_pred, title_ref):
    fig = plt.figure(figsize=(17, 7))

    ### Reference
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')  # First subplot
    nbrs_ref = NearestNeighbors(n_neighbors=num_neighbors).fit(x_ref)
    distances_ref, _ = nbrs_ref.kneighbors(x_ref)
    density_ref = 1 / distances_ref[:, -1]
    density_ref_normalized = (density_ref - density_ref.min()) / (density_ref.max() - density_ref.min())
    plot1 = ax1.scatter(x_ref[:, 0], x_ref[:, 1], x_ref[:, 2], c=density_ref_normalized, cmap="inferno", s=50)
    cb1 = fig.colorbar(plot1, ax=ax1, shrink=0.6)
    cb1.set_label('Density')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(title_ref)

    ### Predicted
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    nbrs_pred = NearestNeighbors(n_neighbors=num_neighbors).fit(x_pred)
    distances_pred, _ = nbrs_pred.kneighbors(x_pred)
    density_pred = 1 / distances_pred[:, -1]
    density_pred_normalized = (density_pred - density_pred.min()) / (density_pred.max() - density_pred.min())
    plot2 = ax2.scatter(x_pred[:, 0], x_pred[:, 1], x_pred[:, 2], c=density_pred_normalized, cmap="inferno", s=50)
    cb2 = fig.colorbar(plot2, ax=ax2, shrink=0.6)
    cb2.set_label('Density')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title(title_pred)

    plt.tight_layout()  # adjust spacing
    plt.show()


def generate_synthetic_data(vae, num_samples, latent_dim, conf_info, conf_encoder):

    conf_info = np.tile(conf_info, (num_samples, 1))
    conf_info = torch.tensor(conf_info, dtype=torch.float32).to(device)

    vae.eval()
    with torch.no_grad():
        dist = torch.distributions.MultivariateNormal(
            torch.zeros(latent_dim, device=device), torch.eye(latent_dim, device=device)
        )

        z_samples = dist.sample((num_samples,))

        conf_info = conf_encoder(conf_info)

        z_samples_concat = torch.cat((z_samples, conf_info), dim=1)

        synthetic_data = vae.decoder(z_samples_concat)
    return synthetic_data


def chamfer_distance(point_cloud1, point_cloud2):
    """
    Compute the Chamfer distance between two point clouds.
    Args:
        point_cloud1 (numpy.ndarray): First point cloud of shape (N, D)
        point_cloud2 (numpy.ndarray): Second point cloud of shape (M, D)
    Returns:
        float: Chamfer distance between the point clouds.
    """
    tree1 = cKDTree(point_cloud1)
    tree2 = cKDTree(point_cloud2)

    # Compute nearest neighbor distances
    distances1, _ = tree1.query(point_cloud2, k=1)
    distances2, _ = tree2.query(point_cloud1, k=1)

    # Average the distances
    chamfer = np.mean(distances1 ** 2) + np.mean(distances2 ** 2)
    return chamfer


def Chamfer_distance_batch(X_pred, X_ref):
    pbar_cd = tqdm(total=X_pred.shape[0], desc="Chamfer distance calculation")
    c_dists = []

    for i in range(X_pred.shape[0]):
        cd = chamfer_distance(X_pred[i], X_ref[i])
        c_dists.append(cd)
        pbar_cd.update()
    pbar_cd.close()

    cd_mean = np.mean(c_dists)
    return cd_mean


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


########################################################################################################################
###### Configurations of the run ######
########################################################################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

batch_size_frac = 0.05
num_dense = 2730
frac_train = 0.9
frac_valid = 0.05
epochs = 300 # 1000
lrate = 4e-4

latent_dimension = 16

NumTrainSamples = int(456 * frac_train)
NumValidSamples = int(456 * frac_valid)
NumTestSamples = 456 - NumTrainSamples - NumValidSamples


########################################################################################################################
###### Data loading & pre processing ######
########################################################################################################################


file_name_cp = "cardiac_atlas_kostas\output\DeterministicAtlas__EstimatedParameters__ControlPoints.txt"
file_name_momenta = "cardiac_atlas_kostas\output\DeterministicAtlas__EstimatedParameters__Momenta.txt"
file_path_confounders = r'C:\Users\kkevopoulos\Downloads\cardiac_function_mesh_complete_final.xlsx'


control_points = load_cp_momenta(file_name_cp)
momenta = load_cp_momenta(file_name_momenta)

x_confounders = pd.read_excel(file_path_confounders)
x_confounders.drop(['BSA', 'Year of birth', 'Height', 'Weight', 'Diastolic BP mean reading',
                 'Systolic BP mean reading', 'mesh_file', 'MAP'], axis=1, inplace=True)
x_confounders['Sex'] = x_confounders['Sex'].replace({'Male': 0, 'Female': 1})
x_confounders = x_confounders.to_numpy()

x_confounders = x_confounders[:, 1:]
expanded_x_confounders = np.tile(x_confounders[:, np.newaxis, :], (1, 2730, 1))



momenta = np.delete(momenta, 0, axis=0)
momenta = momenta.reshape((456, 2730, 3))

density_feature = np.zeros((456, 2730, 1))



for i in range(momenta.shape[0]):
    nbrs = NearestNeighbors(n_neighbors=8).fit(momenta[i])
    distances, _ = nbrs.kneighbors(momenta[i])
    density = 1 / distances[:, -1]
    density_normalized = (density - density.min()) / (density.max() - density.min())

    density_feature[i] = density_normalized.reshape(-1, 1)

momenta = np.concatenate((momenta, density_feature), axis=-1)

X_train_momenta = momenta[:NumTrainSamples, :, :].reshape((NumTrainSamples, 4, 13, 14, 15))
X_valid_momenta = momenta[NumTrainSamples:NumTrainSamples + NumValidSamples, :, :].reshape((NumValidSamples,4, 13, 14, 15))
X_test_momenta = momenta[NumTrainSamples + NumValidSamples:, :, :].reshape((NumTestSamples, 4, 13, 14, 15))


X_train_confounders = minmax_scale(x_confounders[:NumTrainSamples, :], axis=0)
X_valid_confounders = minmax_scale(x_confounders[NumTrainSamples:NumTrainSamples+NumValidSamples, :], axis=0)
X_test_confounders = minmax_scale(x_confounders[NumTrainSamples+NumValidSamples:, :], axis=0)



########################################################################################################################
###### Build the VAE model ######
########################################################################################################################


dataset_train = Data(X=X_train_momenta, y=X_train_confounders)
dataset_valid = Data(X=X_valid_momenta, y=X_valid_confounders)

train_loader = DataLoader(dataset=dataset_train, batch_size=5, shuffle=True)
valid_loader = DataLoader(dataset=dataset_valid, batch_size=int(batch_size_frac * NumValidSamples), shuffle=False)

Convol_vae = ConvVAE(latent_dim=latent_dimension).to(device)

optimizer = torch.optim.Adam(Convol_vae.parameters(), lr=lrate)


########################################################################################################################
###### Train the model ######
########################################################################################################################

trainer = TrainerConvVAE(model=Convol_vae, optimizer=optimizer, epochs=epochs,
                         train_loader=train_loader, valid_loader=valid_loader)

Convol_vae, metadata_encoder = trainer.training()
trainer.plot_losses()

# torch.save(Convol_vae, "model.pth")
# torch.save(Convol_vae.encoder, "encoder.pth")
# torch.save(Convol_vae.decoder, "decoder.pth")
# torch.save(Convol_vae.metadata_enc, "metadata_encoder.pth")

