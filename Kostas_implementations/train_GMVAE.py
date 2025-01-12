from GMVAE import *
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.utils.data as data
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from sklearn.manifold import TSNE


def debug_visuals(dist):
    z_latent = dist.mean.detach().numpy()

    z_embedded = TSNE(n_components=2, learning_rate='auto',
                      init='random', perplexity=7).fit_transform(z_latent)

    plt.scatter(z_embedded[:, 0], z_embedded[:, 1])
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.title("t-SNE visualization of a 2D latent space")
    plt.show()

    for i in range(latent_dimension):
        plt.figure()
        plt.hist(z_latent[:, i], bins=30, color='b', alpha=0.7)
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


############
# Data loading
############

control_points = np.loadtxt(
    "/Users/konstantinoskevopoulos/Downloads/cardiac_atlas_kostas/output/DeterministicAtlas__EstimatedParameters__ControlPoints.txt")

momenta = np.loadtxt(
    "/Users/konstantinoskevopoulos/Downloads/cardiac_atlas_kostas/output/DeterministicAtlas__EstimatedParameters__Momenta.txt")

### TODO: Potentially normalize the momenta here?

momenta = np.delete(momenta, 0, axis=0)
momenta = momenta.reshape((456, 2730, 3))

momenta = torch.tensor(momenta[:, :2704, :], dtype=torch.float32)
momenta = momenta.reshape((456, 3, 52, 52))

batch_size_frac = 0.15
num_dense = 2704
frac_train = 0.9
frac_valid = 0.05
epochs = 4000
lrate = 4e-4
k_diff = 4

latent_dimension = 32

NumTrainSamples = int(456 * frac_train)
NumValidSamples = int(456 * frac_valid)
NumTestSamples = 456 - NumTrainSamples - NumValidSamples

X_train_momenta = momenta[:NumTrainSamples, :, :, :]
X_valid_momenta = momenta[NumTrainSamples:NumTrainSamples + NumValidSamples, :, :, :]
X_test_momenta = momenta[NumTrainSamples + NumValidSamples:, :, :, :]

dataset_train = Data(X=X_train_momenta, y=X_train_momenta)
dataset_valid = Data(X=X_valid_momenta, y=X_valid_momenta)

train_loader = DataLoader(dataset=dataset_train, batch_size=int(batch_size_frac * NumTrainSamples), shuffle=True)
valid_loader = DataLoader(dataset=dataset_valid, batch_size=int(batch_size_frac * NumValidSamples), shuffle=False)

qy_x = Qy_x(k=k_diff, encoder=ConvEncoder(), enc_out_dim=52 * 6 * 6)
qz_xy = Qz_xy(k=k_diff, encoder=ConvEncoder(), enc_out_dim=52 * 6 * 6, hidden_size=1000, latent_dim=latent_dimension)
px_z = Px_z(k=k_diff, decoder=ConvDecoder(latent_dim=latent_dimension))


recon_loss = MSE()
loss = TotalLoss(k_diff, recon_loss)

gmvae = GMVAE(k=k_diff, Qy_x_net=qy_x, Qz_xy_net=qz_xy, Px_z_net=px_z)
optimizer = torch.optim.Adam(gmvae.parameters(), lr=lrate)

trainer = TrainerGMVAE(model=gmvae, optimizer=optimizer, epochs=epochs,
                       train_loader=train_loader, valid_loader=valid_loader, criterion=loss)

gmvae_trained = trainer.training()
trainer.plot_losses()

### Evaluation phase
gmvae_trained.eval()


train, infer = gmvae_trained(X_test_momenta)
reconstruction_test = infer["x_hat"]

reconstruction_test = reconstruction_test.reshape(NumTestSamples, num_dense, 3).detach().numpy()

X_test_momenta = X_test_momenta.reshape(X_test_momenta.shape[0], num_dense, 3).detach().numpy()

data_pred = np.array([control_points[:num_dense, :] + reconstruction_test[i] for i in range(reconstruction_test.shape[0])])
data_ref = np.array([control_points[:num_dense, :] + X_test_momenta[i] for i in range(X_test_momenta.shape[0])])

cd_err_batch = Chamfer_distance_batch(data_pred, data_ref)
print(f"Chamfer distance test: {cd_err_batch}")

for i in range(X_test_momenta.shape[0]):
    visualization(num_neighbors=8, x_pred=data_pred[i], x_ref=data_ref[i], title_pred="Predicted",
                  title_ref="Reference")
