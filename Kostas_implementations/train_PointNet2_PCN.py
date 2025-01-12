import numpy as np
import os
import datetime
from tqdm import tqdm
import pandas as pd
from Encoder_PCN import *
from Decoder_PCN import *
from VAE import *
from torch.utils.data import DataLoader
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from Encoder_PointNet2 import *


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


# def chamfer_distance(p, q):
#     """
#     Computes Chamfer distance between two point clouds.
#
#     Args:
#         p: Tensor of shape (N, D) representing point cloud P.
#         q: Tensor of shape (M, D) representing point cloud Q.
#
#     Returns:
#         Chamfer distance between the two point clouds.
#     """
#     # Compute pairwise squared Euclidean distances between all points in p and q
#     p = p.unsqueeze(1)  # Shape (N, 1, D)
#     q = q.unsqueeze(0)  # Shape (1, M, D)
#
#     dist_matrix = torch.norm(p - q, dim=2) ** 2  # Shape (N, M)
#
#     # Find the minimum distance from each point in p to the points in q
#     min_dist_p_to_q, _ = torch.min(dist_matrix, dim=1)
#
#     # Find the minimum distance from each point in q to the points in p
#     min_dist_q_to_p, _ = torch.min(dist_matrix, dim=0)
#
#     # Compute the Chamfer distance
#     chamfer_dist = torch.mean(min_dist_p_to_q) + torch.mean(min_dist_q_to_p)
#
#     return chamfer_dist


def loss_function(beta_value, dense_prediction, x_pointcloud, logvar, mean):
    ### Loss function
    ### compare prediction with reconstruction "x"
    loss_reconstruct = torch.nn.MSELoss()
    loss_recon = loss_reconstruct(dense_prediction, x_pointcloud.transpose(2, 1))

    loss_kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), 1)
    loss_kl = torch.mean(loss_kl)

    loss_func = loss_recon + beta_value * loss_kl

    return loss_func, loss_recon


def visualization(num_neighbors, x_pred, x_ref, title_pred, title_ref, num_patient, type_batch):
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
    plt.savefig(
        f"/Users/konstantinoskevopoulos/Desktop/Generated_shapes/PointNet2+PCN/Reconstruction/plots/{type_batch}/recon_{num_patient}.png")
    plt.close()

    # plt.figure()
    # difference = (x_ref - x_pred) / x_ref
    #
    # nbrs = NearestNeighbors(n_neighbors=num_neighbors).fit(difference)
    # distances, _ = nbrs.kneighbors(difference)
    # density = 1 / distances[:, -1]
    # density_normalized = (density - density.min()) / (density.max() - density.min())
    #
    # x = difference[:, 0]
    # y = difference[:, 1]
    # z = difference[:, 2]
    #
    # ax = plt.axes(projection='3d')
    # ax.set_title(r"$x_{ref} - x_{pred}$ ")
    # plot = ax.scatter(x, y, z, c=density_normalized, cmap="inferno", s=50)
    # cb = fig.colorbar(plot, ax=ax, shrink=0.6)
    # cb.set_label('Density')


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
        cd = chamfer_distance(X_pred[i].detach().numpy(), X_ref[i])
        c_dists.append(cd)
        pbar_cd.update()
    pbar_cd.close()

    cd_mean = np.mean(c_dists)
    return cd_mean


def results_errors(X_pred, X_ref, type_batch="train"):

    loss = torch.nn.MSELoss()

    loss_batch = loss(X_pred, torch.tensor(X_ref, dtype=torch.float32))
    print(f"MSE {type_batch}: {loss_batch}")

    cd_err_batch = Chamfer_distance_batch(X_pred, X_ref)
    print(f"Chamfer distance {type_batch}: {cd_err_batch}")

    for i in range(X_pred.shape[0]):

        ### Currently I visualise both training and test data
        visualization(num_neighbors=8, x_pred=X_pred[i].detach().numpy(), x_ref=X_ref[i],
                      title_pred=f"Predicted, patient {i}", title_ref=f"Reference, patient {i}",
                      num_patient=i, type_batch=type_batch)

        np.savetxt(f'/Users/konstantinoskevopoulos/Desktop/Generated_shapes/PointNet2+PCN/Reconstruction/'
                   f'rawdata/{type_batch}/reconstructed_{i}.txt', X_pred[i].detach().numpy(), delimiter='\t')


def synthetic_generation(metadata_vector, num_samples, trained_vae):

    generated_shapes = trained_vae.generate(x_metadata=metadata_vector, num_samples=num_samples)

    for i in range(generated_shapes.shape[0]):
        fig = plt.figure(figsize=(8, 8))
        nbrs = NearestNeighbors(n_neighbors=8).fit(generated_shapes[i].detach().numpy())
        ax = plt.axes(projection='3d')
        distances, _ = nbrs.kneighbors(generated_shapes[i].detach().numpy())
        density = 1 / distances[:, -1]
        density_normalized = (density - density.min()) / (density.max() - density.min())
        x = generated_shapes[i, :, 0].detach().numpy()
        y = generated_shapes[i, :, 1].detach().numpy()
        z = generated_shapes[i, :, 2].detach().numpy()
        plot1 = ax.scatter(x, y, z, c=density_normalized, cmap="inferno", s=50)
        cb1 = fig.colorbar(plot1, ax=ax, shrink=0.6)
        cb1.set_label('Density')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f" Synthetic Patient {i} \n \nsex:{metadata_vector[0]} "
                     f"\nBMI: {metadata_vector[1]} "
                     f"\nAge: {metadata_vector[2]}")

        plt.savefig(
            f"/Users/konstantinoskevopoulos/Desktop/Generated_shapes/PointNet2+PCN/Generation/plots/generated_sample_{i}.png")
        plt.close()

        np.savetxt(f'/Users/konstantinoskevopoulos/Desktop/Generated_shapes/PointNet2+PCN/Generation/rawdata/generated_sample_{i}.txt',
                   generated_shapes[i].detach().numpy())


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

data = np.array(
    [control_points + momenta[i] for i in range(momenta.shape[0])])  # Convert list to a single numpy.ndarray
deformed = data
deformed = deformed.transpose((0, 2, 1))


####################
# encoder: PCN
# decoder: PCN
####################

### Dimension of the PCN last maxpool, before the mlp of the pcn
encoder_output_dimension = 512

### Dimension of the latent space
latent_dimension = 16

### Regarding the first and second conv layers of the PCN decoder
conv = [3, 64, 128, 256]

### This mlp is used to produce the output of the PCN encoder
### Dimensions of the mlp to PCN encoder output and bool for dropout
mlp_to_encoder_output = [[encoder_output_dimension, 256, 128, 64, latent_dimension], False]

### This fc defines the architecture of the decoder
### Dimensions of the fcnn layers, and bool for the batch normalization
fc_decoder_configs = [500, 500, 2704, True]

### Initially opt to not encode the metadata vector
encode_metadata = False

### Regarding the PCN decoder
gridsize = 2

### architecture of the mlp from y_encoded --> z latent space
### 3 refers to the dimensionality of the metadata vector
to_latent_params = [latent_dimension + 3, latent_dimension * 2, 2, 23, True]

### architecture of the mlp from z_concatenated_sampled --> y_to_decoder
### mlp that maps the ((z_sample, x_metadata)) to the decoder
### 3 refers to the dimensionality of the metadata vector
to_decoder_params = [latent_dimension + 3, latent_dimension, 2, 23, False]

### Regarding the mlp and final conv layers employed in the PCN decoder]
mlp_configs_pcn_decoder = [250, 256]

### Other training settings
batch_size_frac = 0.029
num_dense = 2704
frac_train = 0.7
frac_valid = 0.15
epochs = 400
lrate = 5e-4
verbose = True

### Annealing of beta --> from AE to VAE. Want to increase the beta relatively early in the training
beta_vae_vals = np.linspace(0.01, 0.4, 5)

encoder = PointNet2_Encoder(num_latent=latent_dimension, additional_channel=False)
decoder = PCN_Decoder(num_dense=num_dense, grid_size=gridsize, configs_mlp=mlp_configs_pcn_decoder)

vae = VAE(encoder=encoder, decoder=decoder, to_latent_params=to_latent_params,
          to_decoder_params=to_decoder_params, decoder_type="PCN")

TrainSamples = int(456 * frac_train)
ValidSamples = int(456 * frac_valid)

X_train_momenta = deformed[:TrainSamples, :, :num_dense]
X_valid_momenta = deformed[TrainSamples:(ValidSamples + TrainSamples), :, :num_dense]
X_test_momenta = deformed[(ValidSamples + TrainSamples):, :, :num_dense]

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

optimizer = torch.optim.Adam(vae.parameters(), lr=lrate)

best_val_loss = float('inf')
best_model_weights = None
mean_loss_train_epochs = []
mean_loss_valid_epochs = []

if verbose:
    pbar = tqdm(total=epochs, desc="Epochs training...")

for epoch in range(epochs):

    ### annealing of hyperparameter alpha. This annealing is still tested

    ### annealing of beta term, used in the loss of the beta-VAE
    ### annealing of beta term, used in the loss of the beta-VAE
    if epoch >= 350:
        beta = beta_vae_vals[4]
    elif epoch >= 300:
        beta = beta_vae_vals[3]
    elif epoch >= 250:
        beta = beta_vae_vals[2]
    elif epoch >= 150:
        beta = beta_vae_vals[1]
    else:
        beta = 0

    vae.train(True)

    total_loss_train = []
    mse_losses = []
    for x_train, y_metadata in train_loader:
        optimizer.zero_grad()

        ### Forward propagation. Inputs to VAE are the batch of point clouds and the patient metadata
        coarse_pred, dense_pred, log_var, mu = vae(x_train, y_metadata)

        loss_train, loss_recon = loss_function(beta_value=beta, dense_prediction=dense_pred,
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
    vae.eval()
    total_loss_valid = []
    mse_losses_valid = []
    with torch.no_grad():
        for x_val, y_val in valid_loader:
            coarse_pred_val, dense_pred_val, log_var_val, mu_val = vae(x_val, y_val)

            loss_valid, loss_recon_valid = loss_function(beta_value=beta, dense_prediction=dense_pred_val,
                                                         logvar=log_var_val, mean=mu_val, x_pointcloud=x_val)
            total_loss_valid.append(loss_valid.item())
            mse_losses_valid.append(loss_recon_valid.detach().numpy())

        mean_loss_valid_epoch = np.mean(total_loss_valid)
        mean_loss_valid_epochs.append(mean_loss_valid_epoch)
        mean_loss_mse_epoch_valid = np.mean(mse_losses_valid)

    ### Keep track of the model that results to the minimum validation error
    if mean_loss_valid_epoch < best_val_loss:
        best_val_loss = mean_loss_valid_epoch
        best_model_weights = vae.state_dict()

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
    plt.axvline(x=150, linestyle='--', color='black')
    plt.axvline(x=250, linestyle='--', color='black')
    plt.axvline(x=300, linestyle='--', color='black')
    plt.axvline(x=350, linestyle='--', color='black')
    plt.xlabel("# Epochs")
    plt.ylabel("Loss function")
    plt.legend()
    plt.savefig(f"/Users/konstantinoskevopoulos/Desktop/Generated_shapes/PointNet2+PCN/training_error.png")
    plt.close()

    plt.figure()
    plt.semilogy(mean_loss_valid_epochs, label='Validation error', color='orange')
    plt.axvline(x=150, linestyle='--', color='black')
    plt.axvline(x=250, linestyle='--', color='black')
    plt.axvline(x=300, linestyle='--', color='black')
    plt.axvline(x=350, linestyle='--', color='black')
    plt.xlabel("# Epochs")
    plt.ylabel("Loss function")
    plt.legend()
    plt.savefig(f"/Users/konstantinoskevopoulos/Desktop/Generated_shapes/PointNet2+PCN/validation_error.png")
    plt.close()

if best_model_weights:
    vae.load_state_dict(best_model_weights)

#################
### Training is over --> evaluate the performance of the model
#################

vae.eval()

_, reconstruction_test, _, _ = vae(torch.tensor(X_test_momenta, dtype=torch.float32),
                                   torch.tensor(X_test_metadata, dtype=torch.float32))

_, reconstruction_train, _, _ = vae(torch.tensor(X_train_momenta, dtype=torch.float32),
                                    torch.tensor(X_train_metadata, dtype=torch.float32))


### Generate synthetic shapes
synthetic_generation(metadata_vector=X_test_metadata[23], num_samples=50, trained_vae=vae)


### Evaluate errors for training dataset
results_errors(X_pred=reconstruction_train, X_ref=X_train_momenta.transpose(0, 2, 1), type_batch="train")
results_errors(X_pred=reconstruction_test, X_ref=X_test_momenta.transpose(0, 2, 1), type_batch="test")
