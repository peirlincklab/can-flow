import torch
import pandas as pd
from sklearn.preprocessing import minmax_scale
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde
import warnings
import shutil
import scipy.stats as stats


warnings.filterwarnings('ignore')


def save_momenta(type_momenta, momenta_vae):
    documents_path = os.path.join(os.environ['USERPROFILE'], 'Documents')
    base_folder = os.path.join(documents_path, f"{type_momenta}_Momenta")

    pbar = tqdm(total=momenta_vae.shape[0], desc="Creating folders and files...")
    for i in range(momenta_vae.shape[0]):
        ### create a folder in Documents called Momenta_i
        subfolder = os.path.join(base_folder, f"Shooting_Momenta_{i}")
        data_folder = os.path.join(subfolder, "data")

        os.makedirs(data_folder, exist_ok=True)

        ###For control points
        write_cp_momenta(base_folder=data_folder, array_towrite=control_points, type="ControlPoints")

        ###For momenta (same name of different files, but different momenta are in different files. Just name is the same)
        write_cp_momenta(base_folder=data_folder, array_towrite=momenta_vae[i], type="Momenta")

        vtk_file = os.path.join(documents_path, "template.vtk")
        destination_file = os.path.join(data_folder, "template.vtk")

        shutil.copy(vtk_file, destination_file)

        model_file = os.path.join(documents_path, 'model.xml')
        destination_file = os.path.join(subfolder, 'model.xml')

        shutil.copy(model_file, destination_file)

        pbar.update()
    pbar.close()


def write_cp_momenta(base_folder, array_towrite, type):
    file_path = os.path.join(base_folder, f"{type}.txt")

    if type=="Momenta":
        # Open the file and write custom content
        with open(file_path, 'w') as file:
            # Write the first row
            file.write(f"1 2730 3\n")
            # Write an empty row
            file.write("\n")

    # Append the numpy array with tab-delimited values
    with open(file_path, 'a') as file:
        np.savetxt(file, array_towrite, delimiter=' ')

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



def debug_visuals(dist):
    z_latent = dist.mean.detach().cpu().numpy()

    z_embedded = TSNE(n_components=2, learning_rate='auto',
                      init='random', perplexity=15).fit_transform(z_latent)

    plt.scatter(z_embedded[:, 0], z_embedded[:, 1])
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.title("t-SNE visualization of a 2D latent space")
    plt.show()

    supports = np.zeros(latent_dimension)
    for i in range(latent_dimension):
        kde = gaussian_kde(z_latent[:, i])
        x = np.linspace(min(z_latent[:, i]), max(z_latent[:, i]), z_latent[:, i].shape[0])
        density = kde(x)

        plt.figure()
        plt.hist(z_latent[:, i], bins=25, color='b', density=True, alpha=0.7, label="Histogram")
        plt.plot(x, density, label="KDE", color="red")
        plt.title(fr"z_{i}")
        plt.legend()
        plt.show()

        support = np.unique(z_latent[:, i])
        support_length = np.max(support) - np.min(support)

        supports[i] = support_length

    print(supports)



    plt.figure()
    for j in range(latent_dimension):
        kde = gaussian_kde(z_latent[:, j])
        x = np.linspace(min(z_latent[:, j]), max(z_latent[:, j]), z_latent[:, j].shape[0])
        density = kde(x)
        plt.plot(x, density, color='blue')
        plt.fill_between(x, density, color='lightblue', alpha=0.6)
    plt.xlabel(r'$z_i$')
    plt.ylabel('PDF')
    plt.show()



def load_cp_momenta(file_name):

    # Get the path to the Downloads folder dynamically
    downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")

    file_path = os.path.join(downloads_folder, file_name)
    data = np.loadtxt(file_path)

    return data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


num_dense = 2730
frac_train = 0.9
frac_valid = 0.05

latent_dimension = 16


NumTrainSamples = int(456 * frac_train)
NumValidSamples = int(456 * frac_valid)
NumTestSamples = 456 - NumTrainSamples - NumValidSamples

NumSynthetic = 20


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
X_valid_momenta = momenta[NumTrainSamples:NumTrainSamples + NumValidSamples, :, :].reshape((NumValidSamples, 4, 13, 14, 15))
X_test_momenta = momenta[NumTrainSamples + NumValidSamples:, :, :].reshape((NumTestSamples, 4, 13, 14, 15))



X_train_confounders = minmax_scale(x_confounders[:NumTrainSamples, 1:], axis=0)
X_valid_confounders = minmax_scale(x_confounders[NumTrainSamples:NumTrainSamples+NumValidSamples, 1:], axis=0)
X_test_confounders = minmax_scale(x_confounders[NumTrainSamples+NumValidSamples:, 1:], axis=0)

model = torch.load("model.pth")
metadata_encoder = torch.load("metadata_encoder.pth")

x_test_confounder_female = [1.0, 0.52473, 0.19999]
x_test_confounder_male = [0.0, 0.52473, 0.19999]

model.eval()
metadata_encoder.eval()

x_train = torch.tensor(X_train_momenta, dtype=torch.float32).to(device)
x_train_conf = torch.tensor(X_train_confounders, dtype=torch.float32).to(device)


x_valid = torch.tensor(X_valid_momenta, dtype=torch.float32).to(device)
x_valid_conf = torch.tensor(X_valid_confounders, dtype=torch.float32).to(device)

x_test = torch.tensor(X_test_momenta, dtype=torch.float32).to(device)
x_test_conf = torch.tensor(X_test_confounders, dtype=torch.float32).to(device)


synthetic_momenta = generate_synthetic_data(vae=model, latent_dim=latent_dimension, num_samples=NumSynthetic,
                                            conf_info=x_test_confounder_female, conf_encoder=metadata_encoder)



reconstruction_train, dist_train, _ = model(x_train, x_train_conf)
reconstruction_valid, _, _ = model(x_valid, x_valid_conf)
reconstruction_test, _, _ = model(x_test, x_test_conf)

reconstruction_train = reconstruction_train.reshape(NumTrainSamples, num_dense, 4).detach().cpu().numpy()
reconstruction_train = reconstruction_train[:, :, :3]


reconstruction_valid = reconstruction_valid.reshape(NumValidSamples, num_dense, 4).detach().cpu().numpy()
reconstruction_valid = reconstruction_valid[:, :, :3]


reconstruction_test = reconstruction_test.reshape(NumTestSamples, num_dense, 4).detach().cpu().numpy()
reconstruction_test = reconstruction_test[:, :, :3]

synthetic_momenta = synthetic_momenta.reshape(NumSynthetic, num_dense, 4).detach().cpu().numpy()
synthetic_momenta = synthetic_momenta[:, :, :3]


reconstructed_momenta = np.concatenate((reconstruction_train, reconstruction_valid, reconstruction_test), axis=0)


# save_momenta(type_momenta="Predicted", momenta_vae=reconstructed_momenta)
save_momenta(type_momenta="Generated_female", momenta_vae=synthetic_momenta)


debug_visuals(dist_train)


X_test_momenta = X_test_momenta.reshape(NumTestSamples, num_dense, 4)
X_test_momenta = X_test_momenta[:, :, :3]

X_test_deformed = np.array([control_points[:num_dense, :] + X_test_momenta[i] for i in range(X_test_momenta.shape[0])])
reconstruction_deformed = np.array([control_points[:num_dense, :] + reconstruction_test[i]
                                    for i in range(reconstruction_test.shape[0])])

cd_err_batch = Chamfer_distance_batch(reconstruction_deformed, X_test_deformed)
print(f"Chamfer distance test: {cd_err_batch}")



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
    ax.set_title(f" Synthetic Patient {i} \n \nsex:{x_test_confounder_female[0]} "
                 f"\nBMI: {x_test_confounder_female[1]} "
                 f"\nAge: {x_test_confounder_female[2]}")

    plt.show()

for i in range(X_test_deformed.shape[0]):
    visualization(num_neighbors=8, x_pred=reconstruction_deformed[i], x_ref=X_test_deformed[i], title_pred="Predicted",
                  title_ref="Reference")






