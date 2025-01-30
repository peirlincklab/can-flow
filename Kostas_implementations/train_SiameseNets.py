import torch
from SiameseNets import *
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.utils.data as data
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from sklearn.manifold import TSNE
import os


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


def generate_synthetic_data(vae, num_samples, latent_dim, ensemble=False):
    vae.eval()
    with torch.no_grad():
        dist = torch.distributions.MultivariateNormal(
            torch.zeros(latent_dim, device=device), torch.eye(latent_dim, device=device)
        )

        z_samples = dist.sample((num_samples,))
        z_samples = z_samples.to(device)
        if ensemble:
            synthetic_data = vae.generation(z_samples)
        else:
            synthetic_data = vae.decoder(z_samples)
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
    chamfer = np.mean(distances1 * 2) + np.mean(distances2 * 2)
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


############
# Configurations of the run
############

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

batch_size_frac = 0.4 ### 0.15
num_dense = 2730
frac_train = 0.9
frac_valid = 0.05
epochs = 500 ### 2000
lrate = 4e-4

latent_dimension = 32

NumTrainSamples = int(456 * frac_train)
NumValidSamples = int(456 * frac_valid)
NumTestSamples = 456 - NumTrainSamples - NumValidSamples


############
# Data loading
############


file_name_cp = "cardiac_atlas_kostas\output\DeterministicAtlas__EstimatedParameters__ControlPoints.txt"

file_name_momenta = "cardiac_atlas_kostas\output\DeterministicAtlas__EstimatedParameters__Momenta.txt"

control_points = load_cp_momenta(file_name_cp)
momenta = load_cp_momenta(file_name_momenta)


momenta = np.delete(momenta, 0, axis=0)
momenta = momenta.reshape((456, 2730, 3))

X_train_momenta = momenta[:NumTrainSamples, :, :]
X_valid_momenta = momenta[NumTrainSamples:NumTrainSamples + NumValidSamples, :, :]
X_test_momenta = momenta[NumTrainSamples + NumValidSamples:, :, :]

dataset_train = Data(X=X_train_momenta, y=X_train_momenta)
dataset_valid = Data(X=X_valid_momenta, y=X_valid_momenta)

train_loader = DataLoader(dataset=dataset_train, batch_size=int(batch_size_frac * NumTrainSamples), shuffle=True)
valid_loader = DataLoader(dataset=dataset_valid, batch_size=int(batch_size_frac * NumValidSamples), shuffle=False)


siamese_vae = SiameseVAE(latent_dim=latent_dimension).to(device)

optimizer = torch.optim.Adam(siamese_vae.parameters(), lr=lrate)

trainer = TrainerSiamese(model=siamese_vae, optimizer=optimizer, epochs=epochs,
                         train_loader=train_loader, valid_loader=valid_loader)

model = trainer.training(anneal=False, beta=5e-3)


model.eval()

X_train_momenta = torch.tensor(X_train_momenta, dtype=torch.float32, device=device)
reconstruction_train, dist_train = model(X_train_momenta)
debug_visuals(dist_train)

reconstruction_train = reconstruction_train.reshape(NumTrainSamples, num_dense, 3).detach().cpu().numpy()


## Denormalize the reconstructed deformed control points of the training set
X_test_momenta = torch.tensor(X_test_momenta, dtype=torch.float32, device=device)
reconstruction_test, dist_test = model(X_test_momenta)
reconstruction_test = reconstruction_test.reshape(NumTestSamples, num_dense, 3).detach().cpu().numpy()

X_test_momenta = X_test_momenta.reshape(NumTestSamples, num_dense, 3).detach().cpu().numpy()


X_test_deformed = np.array([control_points[:num_dense, :] + X_test_momenta[i] for i in range(X_test_momenta.shape[0])])
reconstruction_deformed = np.array([control_points[:num_dense, :] + reconstruction_test[i]
                                    for i in range(reconstruction_test.shape[0])])


cd_err_batch = Chamfer_distance_batch(reconstruction_deformed, X_test_deformed)
print(f"Chamfer distance test: {cd_err_batch}")

synthetic_momenta = generate_synthetic_data(vae=model, latent_dim=latent_dimension, num_samples=10)
synthetic_momenta = synthetic_momenta.reshape(10, num_dense, 3).detach().cpu().numpy()

synthetic_shapes = np.array([control_points[:num_dense, :] + synthetic_momenta[i] for i in range(synthetic_momenta.shape[0])])


for i in range(synthetic_shapes.shape[0]):
    fig = plt.figure(figsize=(8, 8))
    nbrs = NearestNeighbors(n_neighbors=8).fit(synthetic_shapes[i])
    ax = plt.axes(projection='3d')
    distances, _ = nbrs.kneighbors(synthetic_shapes[i])
    density = 1 / distances[:, -1]
    density_normalized = (density - density.min()) / (density.max() - density.min())
    x = synthetic_shapes[i, :, 0]
    y = synthetic_shapes[i, :, 1]
    z = synthetic_shapes[i, :, 2]
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

for i in range(X_test_deformed.shape[0]):
    visualization(num_neighbors=8, x_pred=reconstruction_deformed[i], x_ref=X_test_deformed[i], title_pred="Predicted",
                  title_ref="Reference")








































# siamese_vaes = []
# ############
# # 1st VAE: beta = 0.01
# ############
#
# siamese_vae = SiameseVAE(latent_dim=latent_dimension).to(device)
#
# optimizer = torch.optim.Adam(siamese_vae.parameters(), lr=lrate)
#
# trainer = TrainerSiamese(model=siamese_vae, optimizer=optimizer, epochs=epochs,
#                          train_loader=train_loader, valid_loader=valid_loader)
#
# siamese_vaes.append(trainer.training(anneal=False, beta=5e-3))
#
#
# ############
# # 2nd VAE: beta = 0.02
# ############
#
#
# siamese_vae = SiameseVAE(latent_dim=latent_dimension).to(device)
#
# optimizer = torch.optim.Adam(siamese_vae.parameters(), lr=lrate)
#
# trainer = TrainerSiamese(model=siamese_vae, optimizer=optimizer, epochs=epochs,
#                          train_loader=train_loader, valid_loader=valid_loader)
#
# siamese_vaes.append(trainer.training(anneal=False, beta=5e-3))
#
#
# ############
# # 3rd VAE: beta = 4e-3
# ############
#
# siamese_vae = SiameseVAE(latent_dim=latent_dimension).to(device)
#
# optimizer = torch.optim.Adam(siamese_vae.parameters(), lr=lrate)
#
# trainer = TrainerSiamese(model=siamese_vae, optimizer=optimizer, epochs=epochs,
#                          train_loader=train_loader, valid_loader=valid_loader)
#
# siamese_vaes.append(trainer.training(anneal=False, beta=5e-3))
#
# ############
# # 4th VAE: beta = annealing
# ############
#
# siamese_vae = SiameseVAE(latent_dim=latent_dimension).to(device)
#
# optimizer = torch.optim.Adam(siamese_vae.parameters(), lr=lrate)
#
# trainer = TrainerSiamese(model=siamese_vae, optimizer=optimizer, epochs=epochs,
#                          train_loader=train_loader, valid_loader=valid_loader)
#
# siamese_vaes.append(trainer.training(anneal=False, beta=5e-3))
#
#
# ############
# # Optimize the ensemble weights
# ############
#
# ensemble_siamese = EnsembleVAE(vae_models=siamese_vaes).to(device)
#
# optimizer_w = torch.optim.Adam(ensemble_siamese.parameters(), lr=0.004)
#
# trainer_ensemble = TrainerEnsembleVAE(model=ensemble_siamese, epochs=epochs, optimizer=optimizer_w,
#                                       train_loader=train_loader, valid_loader=valid_loader)
#
# ensemble_siamese = trainer_ensemble.training(anneal=False, beta=5e-3)
# trainer_ensemble.plot_losses()
#
#
# ### Evaluation phase
# ensemble_siamese.eval()
#
# reconstruction_train, dist_train = ensemble_siamese(torch.tensor(X_train_momenta, dtype=torch.float32))
# debug_visuals(dist_train)
#
# reconstruction_train = reconstruction_train.reshape(NumTrainSamples, num_dense, 3).detach().cpu().numpy()
#
#
# ## Denormalize the reconstructed deformed control points of the training set
# reconstruction_test, dist_test = ensemble_siamese(torch.tensor(X_test_momenta, dtype=torch.float32))
# reconstruction_test = reconstruction_test.reshape(NumTestSamples, num_dense, 3).detach().cpu().numpy()
#
# X_test_momenta = X_test_momenta.reshape(NumTestSamples, num_dense, 3)
#
#
# X_test_deformed = np.array([control_points[:num_dense, :] + X_test_momenta[i] for i in range(X_test_momenta.shape[0])])
# reconstruction_deformed = np.array([control_points[:num_dense, :] + reconstruction_test[i]
#                                     for i in range(reconstruction_test.shape[0])])
#
#
# cd_err_batch = Chamfer_distance_batch(reconstruction_deformed, X_test_deformed)
# print(f"Chamfer distance test: {cd_err_batch}")
#
# synthetic_momenta = generate_synthetic_data(vae=ensemble_siamese, latent_dim=latent_dimension, num_samples=10, ensemble=True)
# synthetic_momenta = synthetic_momenta.reshape(10, num_dense, 3).detach().cpu().numpy()
#
# synthetic_shapes = np.array([control_points[:num_dense, :] + synthetic_momenta[i] for i in range(synthetic_momenta.shape[0])])
#
#
# for i in range(synthetic_shapes.shape[0]):
#     fig = plt.figure(figsize=(8, 8))
#     nbrs = NearestNeighbors(n_neighbors=8).fit(synthetic_shapes[i])
#     ax = plt.axes(projection='3d')
#     distances, _ = nbrs.kneighbors(synthetic_shapes[i])
#     density = 1 / distances[:, -1]
#     density_normalized = (density - density.min()) / (density.max() - density.min())
#     x = synthetic_shapes[i, :, 0]
#     y = synthetic_shapes[i, :, 1]
#     z = synthetic_shapes[i, :, 2]
#     plot1 = ax.scatter(x, y, z, c=density_normalized, cmap="inferno", s=50)
#     cb1 = fig.colorbar(plot1, ax=ax, shrink=0.6)
#     cb1.set_label('Density')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     # ax.set_title(f" Synthetic Patient {i} \n \nsex:{metadata_vector[0]} "
#     #              f"\nBMI: {metadata_vector[1]} "
#     #              f"\nAge: {metadata_vector[2]}")
#
#     plt.show()
#
# for i in range(X_test_deformed.shape[0]):
#     visualization(num_neighbors=8, x_pred=reconstruction_deformed[i], x_ref=X_test_deformed[i], title_pred="Predicted",
#                   title_ref="Reference")