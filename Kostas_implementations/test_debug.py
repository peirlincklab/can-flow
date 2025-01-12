import warnings
import os
import torch
from Encoder_2DConv import *
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
import torch.utils.data as data
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from sklearn.preprocessing import minmax_scale

warnings.filterwarnings('ignore')


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


############
# Data loading
############

### TODO: Check the information from patient metadata (ordering)
x_metadata = pd.read_excel('/Users/konstantinoskevopoulos/Downloads/cardiac_function_mesh_complete_final.xlsx')
x_metadata.drop(['BSA', 'Year of birth', 'Height', 'Weight', 'Diastolic BP mean reading',
                 'Systolic BP mean reading', 'mesh_file', 'MAP'], axis=1, inplace=True)
x_metadata['Sex'] = x_metadata['Sex'].replace({'Male': 0, 'Female': 1})
x_metadata = x_metadata.to_numpy()

x_metadata = minmax_scale(x_metadata, axis=0)

############
# Data preprocessing
############

control_points = np.loadtxt(
    "/Users/konstantinoskevopoulos/Downloads/cardiac_atlas_kostas/output/DeterministicAtlas__EstimatedParameters__ControlPoints.txt")

momenta = np.loadtxt(
    "/Users/konstantinoskevopoulos/Downloads/cardiac_atlas_kostas/output/DeterministicAtlas__EstimatedParameters__Momenta.txt")

momenta = np.delete(momenta, 0, axis=0)
momenta = momenta.reshape((456, 2730, 3))

momenta = torch.tensor(momenta[:, :2704, :], dtype=torch.float32)

momenta = momenta.reshape((456, 3, 52, 52))

# permutation = torch.randperm(momenta.size(0))
#
# # Apply the permutation to the tensor along the first dimension
# momenta = momenta[permutation]

############
# Training settings CVAE
############

batch_size_frac = 0.15
num_dense = 2704
frac_train = 0.9
frac_valid = 0.05
epochs = 1000
lrate = 5e-4


cvae = CVAE()

TrainSamples = int(456 * frac_train)
ValidSamples = int(456 * frac_valid)
TestSamples = 456 - TrainSamples - ValidSamples

X_train_momenta = momenta[:TrainSamples, :, :, :]
X_valid_momenta = momenta[TrainSamples:(TrainSamples + ValidSamples), :, :, :]
X_test_momenta = momenta[(TrainSamples + ValidSamples):, :, :, :]

X_train_metadata = x_metadata[:TrainSamples, 1:]
X_valid_metadata = x_metadata[TrainSamples:(ValidSamples + TrainSamples), 1:]
X_test_metadata = x_metadata[(ValidSamples + TrainSamples):, 1:]

### In this case y = x_momenta because the VAE aims to reconstruct the input
### So, x=y (momenta point clouds).
### For ease of implementation, we give as y input to the "Data" class the x_metadata information
dataset_train = Data(X=X_train_momenta, y=X_train_metadata)
dataset_valid = Data(X=X_valid_momenta, y=X_valid_metadata)

train_loader = DataLoader(dataset=dataset_train, batch_size=int(batch_size_frac * TrainSamples), shuffle=True)
valid_loader = DataLoader(dataset=dataset_valid, batch_size=int(batch_size_frac * ValidSamples), shuffle=True)

optimizer = torch.optim.Adam(cvae.parameters(), lr=lrate)

############
# First train CVAE
############

cvae_trainer = CVAETrainer(model=cvae, optimizer=optimizer, epochs=epochs, train_loader=train_loader,
                           valid_loader=valid_loader)

cvae = cvae_trainer.training()
cvae_trainer.plot_losses()


### Evaluation phase
cvae.eval()

reconstruction_test, _, _ = cvae(torch.tensor(X_test_momenta, dtype=torch.float32))

reconstruction_test = reconstruction_test.reshape(TestSamples, num_dense, 3).detach().numpy()


X_test_momenta = X_test_momenta.reshape(X_test_momenta.shape[0], num_dense, 3).detach().numpy()

data_pred = np.array([control_points[:num_dense, :] + reconstruction_test[i] for i in range(X_test_momenta.shape[0])])
data_ref = np.array([control_points[:num_dense, :] + X_test_momenta[i] for i in range(X_test_momenta.shape[0])])

cd_err_batch = Chamfer_distance_batch(data_pred, data_ref)
print(f"Chamfer distance test: {cd_err_batch}")

# gen_shapes = cvae.generate(x_metadata=None, num_samples=10)
num_samples = 10

cvae.decoder.eval()
cvae.to_decoder_input.eval()
with torch.no_grad():
    z_samples = torch.randn(num_samples, 32)

    # y_to_decoder = torch.cat((z_samples, x_metadata_generate), dim=1)
    y_to_decoder = cvae.to_decoder_input(z_samples).reshape((num_samples, 512, 6, 6))

    gen_shapes = cvae.decoder(y_to_decoder)


gen_shapes = gen_shapes.reshape(gen_shapes.shape[0], num_dense, 3).detach().numpy()

data_gen = np.array([control_points[:num_dense, :] + gen_shapes[i] for i in range(gen_shapes.shape[0])])

for i in range(gen_shapes.shape[0]):
    fig = plt.figure(figsize=(8, 8))
    nbrs = NearestNeighbors(n_neighbors=8).fit(gen_shapes[i])
    ax = plt.axes(projection='3d')
    distances, _ = nbrs.kneighbors(gen_shapes[i])
    density = 1 / distances[:, -1]
    density_normalized = (density - density.min()) / (density.max() - density.min())
    x = data_gen[i, :, 0]
    y = data_gen[i, :, 1]
    z = data_gen[i, :, 2]
    plot1 = ax.scatter(x, y, z, c=density_normalized, cmap="inferno", s=50)
    cb1 = fig.colorbar(plot1, ax=ax, shrink=0.6)
    cb1.set_label('Density')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_title(f" Synthetic Patient {i} \n \nsex:{metadata_vector[0]} "
    #              f"\nBMI: {metadata_vector[1]} "
    #              f"\nAge: {metadata_vector[2]}")

    plt.show()

for i in range(X_test_momenta.shape[0]):
    visualization(num_neighbors=8, x_pred=data_pred[i], x_ref=data_ref[i], title_pred="Predicted",
                  title_ref="Reference")

