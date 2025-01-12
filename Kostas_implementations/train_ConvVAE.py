import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from Encoder_2DConv import *
from torch.utils.data import DataLoader
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree


def synthetic_generation(metadata_vector, num_samples, trained_cvae):

    generated_shapes = trained_cvae.generate(x_metadata=metadata_vector, num_samples=num_samples)

    generated_shapes = generated_shapes.reshape(generated_shapes.shape[0], num_dense, 3).detach().numpy()

    data_gen = np.array([control_points[:num_dense, :] + generated_shapes[i] for i in range(generated_shapes.shape[0])])

    for i in range(generated_shapes.shape[0]):
        fig = plt.figure(figsize=(8, 8))
        nbrs = NearestNeighbors(n_neighbors=8).fit(generated_shapes[i])
        ax = plt.axes(projection='3d')
        distances, _ = nbrs.kneighbors(generated_shapes[i])
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
        ax.set_title(f" Synthetic Patient {i} \n \nsex:{metadata_vector[0]} "
                     f"\nBMI: {metadata_vector[1]} "
                     f"\nAge: {metadata_vector[2]}")

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


def loss_function(beta_value, dense_prediction, x_pointcloud, logvar, mean):
    ### compare prediction with reconstruction "x"
    loss_reconstruct = torch.nn.MSELoss()
    loss_recon_ = loss_reconstruct(dense_prediction, x_pointcloud)

    loss_kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), 1)
    loss_kl = torch.mean(loss_kl)

    loss_func = loss_recon_ + beta_value * loss_kl

    return loss_func, loss_recon_


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


### TODO: Check the information from patient metadata (ordering)
x_metadata = pd.read_excel('/Users/konstantinoskevopoulos/Downloads/cardiac_function_mesh_complete_final.xlsx')
x_metadata.drop(['BSA', 'Year of birth', 'Height', 'Weight', 'Diastolic BP mean reading',
                 'Systolic BP mean reading', 'mesh_file', 'MAP'], axis=1, inplace=True)
x_metadata['Sex'] = x_metadata['Sex'].replace({'Male': 0, 'Female': 1})
x_metadata = x_metadata.to_numpy()

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

# ### Training settings
# batch_size_frac = 0.4
# num_dense = 2704
# frac_train = 0.7
# frac_valid = 0.15
# epochs = 2
# lrate = 5e-4
# verbose = True
#
# ### Schedule of beta annealing
# ep1 = 250
# ep2 = 400
# ep3 = 500
# ep4 = 600
#
# ### Annealing of beta --> from AE to VAE. Want to increase the beta relatively early in the training
# beta_vae_vals = np.linspace(0.05, 0.5, 5)
#
# cvae = CVAE(enc_metadata=False)
#
# TrainSamples = int(456 * frac_train)
# ValidSamples = int(456 * frac_valid)
# TestSamples = 456 - TrainSamples - ValidSamples
#
# X_train_momenta = momenta[:TrainSamples, :, :, :]
# X_valid_momenta = momenta[TrainSamples:(TrainSamples + ValidSamples), :, :, :]
# X_test_momenta = momenta[(TrainSamples + ValidSamples):, :, :, :]
#
# X_train_metadata = x_metadata[:TrainSamples, 1:]
# X_valid_metadata = x_metadata[TrainSamples:(ValidSamples + TrainSamples), 1:]
# X_test_metadata = x_metadata[(ValidSamples + TrainSamples):, 1:]
#
# ### In this case y = x_momenta because the VAE aims to reconstruct the input
# ### So, x=y (momenta point clouds).
# ### For ease of implementation, we give as y input to the "Data" class the x_metadata information
# dataset_train = Data(X=X_train_momenta, y=X_train_metadata)
# dataset_valid = Data(X=X_valid_momenta, y=X_valid_metadata)
#
# train_loader = DataLoader(dataset=dataset_train, batch_size=int(batch_size_frac * TrainSamples), shuffle=True)
# valid_loader = DataLoader(dataset=dataset_valid, batch_size=int(batch_size_frac * ValidSamples), shuffle=True)
#
# optimizer = torch.optim.Adam(cvae.parameters(), lr=lrate)






best_val_loss = float('inf')
best_model_weights = None
mean_loss_train_epochs = []
mean_loss_valid_epochs = []

if verbose:
    pbar = tqdm(total=epochs, desc="Epochs training...")

for epoch in range(epochs):

    ### annealing of hyperparameter alpha. This annealing is still tested
    ### annealing of beta term, used in the loss of the beta-VAE
    if epoch >= ep_vals[3]:
        beta = beta_vae_vals[4]
    elif epoch >= ep_vals[2]:
        beta = beta_vae_vals[3]
    elif epoch >= ep_vals[1]:
        beta = beta_vae_vals[2]
    elif epoch >= ep_vals[0]:
        beta = beta_vae_vals[1]
    else:
        beta = 0

    cvae.train(True)

    total_loss_train = []
    mse_losses = []
    for x_train, y_metadata in train_loader:
        optimizer.zero_grad()

        ### Forward propagation. Inputs to VAE are the batch of point clouds and the patient metadata
        prediction, log_var, mu = cvae(x_train, y_metadata)

        loss_train, loss_recon = loss_function(beta_value=beta, dense_prediction=prediction,
                                               logvar=log_var, mean=mu, x_pointcloud=x_train)
        ### total loss can be divided by the len dataset (because this is the batch loss)
        total_loss_train.append(loss_train.item())
        mse_losses.append(loss_recon.detach().numpy())

        ### Backprop
        loss_train.backward()
        optimizer.step()

    mean_loss_train_epoch = np.mean(total_loss_train)
    mean_loss_train_epochs.append(mean_loss_train_epoch)
    mean_loss_mse_epoch = np.mean(mse_losses)

    ### Validation phase
    cvae.eval()
    total_loss_valid = []
    mse_losses_valid = []
    with torch.no_grad():
        for x_val, y_val in valid_loader:
            prediction_val, log_var_val, mu_val = cvae(x_val, y_val)

            loss_valid, loss_recon_valid = loss_function(beta_value=beta, dense_prediction=prediction_val,
                                                         logvar=log_var_val, mean=mu_val, x_pointcloud=x_val)
            total_loss_valid.append(loss_valid.item())
            mse_losses_valid.append(loss_recon_valid.detach().numpy())

        mean_loss_valid_epoch = np.mean(total_loss_valid)
        mean_loss_valid_epochs.append(mean_loss_valid_epoch)
        mean_loss_mse_epoch_valid = np.mean(mse_losses_valid)

    ### Keep track of the model that results to the minimum validation error
    if mean_loss_valid_epoch < best_val_loss and epoch > ep1:
        best_val_loss = mean_loss_valid_epoch
        best_model_weights = cvae.state_dict()

    if verbose:
        print(f"\nEpoch   Training   Validation    Validation MSE    Beta\n"
              f"{epoch}   {mean_loss_train_epoch}   {mean_loss_valid_epoch}    {mean_loss_mse_epoch_valid}    {beta}\n"
              f"====================================================")
    if verbose:
        pbar.update()
if verbose:
    pbar.close()
    print("Done training!")

if verbose:
    ### Plot the losses
    plt.figure()
    plt.semilogy(mean_loss_train_epochs, label='Training error')
    plt.axvline(x=ep_vals[0], linestyle='--', color='black')
    plt.axvline(x=ep_vals[1], linestyle='--', color='black')
    plt.axvline(x=ep_vals[2], linestyle='--', color='black')
    plt.axvline(x=ep_vals[3], linestyle='--', color='black')
    plt.xlabel("# Epochs")
    plt.ylabel("Loss function")
    plt.legend()
    plt.show()

    plt.figure()
    plt.semilogy(mean_loss_valid_epochs, label='Validation error', color='orange')
    plt.axvline(x=ep_vals[0], linestyle='--', color='black')
    plt.axvline(x=ep_vals[1], linestyle='--', color='black')
    plt.axvline(x=ep_vals[2], linestyle='--', color='black')
    plt.axvline(x=ep_vals[3], linestyle='--', color='black')
    plt.xlabel("# Epochs")
    plt.ylabel("Loss function")
    plt.legend()
    plt.show()

if best_model_weights:
    cvae.load_state_dict(best_model_weights)

#################
### Training is over --> evaluate the performance of the model
#################

cvae.eval()

reconstruction_test, _, _ = cvae(torch.tensor(X_test_momenta, dtype=torch.float32),
                                 torch.tensor(X_test_metadata, dtype=torch.float32))


reconstruction_test = reconstruction_test.reshape(TestSamples, num_dense, 3).detach().numpy()
X_test_momenta = X_test_momenta.reshape(X_test_momenta.shape[0], num_dense, 3).detach().numpy()


data_pred = np.array([control_points[:num_dense, :] + reconstruction_test[i] for i in range(X_test_momenta.shape[0])])
data_ref = np.array([control_points[:num_dense, :] + X_test_momenta[i] for i in range(X_test_momenta.shape[0])])


cd_err_batch = Chamfer_distance_batch(data_pred, data_ref)
print(f"Chamfer distance test: {cd_err_batch}")

synthetic_generation(metadata_vector=X_test_metadata[23], num_samples=20, trained_cvae=cvae)

for i in range(X_test_momenta.shape[0]):
    visualization(num_neighbors=8, x_pred=data_pred[i], x_ref=data_ref[i], title_pred="Predicted", title_ref="Reference")



