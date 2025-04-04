import torch
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
from scipy.stats import wasserstein_distance




def load_cp_momenta(file_name):

    # Get the path to the Downloads folder dynamically
    downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")

    file_path = os.path.join(downloads_folder, file_name)
    data = np.loadtxt(file_path)

    return data



latent_dimension = 80

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


file_name_cp = r"cardiac_atlas_kostas\output\DeterministicAtlas__EstimatedParameters__ControlPoints.txt"
file_name_momenta = r"cardiac_atlas_kostas\output\DeterministicAtlas__EstimatedParameters__Momenta.txt"

control_points = load_cp_momenta(file_name_cp)
momenta = load_cp_momenta(file_name_momenta)
momenta = np.delete(momenta, 0, axis=0)
momenta = momenta.reshape((456, 2730, 3))
momenta = momenta.reshape((456, 3, 13, 14, 15))
momenta = torch.tensor(momenta, dtype=torch.float32).to(device)

encoder = torch.load("../data_models_saved/encoder.pth")
decoder = torch.load("../data_models_saved/decoder.pth")


synthetic_z = np.load("../data_models_saved/synthetic_z.npy")
synthetic_z = torch.tensor(synthetic_z, dtype=torch.float32).to(device)

real_z = encoder(momenta)

synthetic_shapes = decoder(synthetic_z)
synthetic_shapes = synthetic_shapes.detach().cpu().numpy()
synthetic_shapes = synthetic_shapes.reshape(synthetic_shapes.shape[0], 2730, 3)


real_z = real_z.detach().cpu().numpy()
synthetic_z = synthetic_z.detach().cpu().numpy()

z = np.concatenate((real_z, synthetic_z))

z_tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=20).fit_transform(z)
plt.figure()
plt.scatter(z_tsne[:456, 0], z_tsne[:456, 1], color='green', label='real')
plt.scatter(z_tsne[456:, 0], z_tsne[456:, 1], color='orange', label='gen')
plt.legend()
plt.title('t-SNE visualization of latent space')
plt.show()

w_dists = []
for i in range(real_z.shape[1]):
    ref_dist = real_z[:, i]
    test_dist = synthetic_z[:, i]


    kde_reference = gaussian_kde(ref_dist)
    x_reference = np.linspace(min(ref_dist), max(ref_dist), ref_dist.shape[0])
    density_reference = kde_reference(x_reference)

    kde_test = gaussian_kde(test_dist)
    x_test = np.linspace(min(test_dist), max(test_dist), test_dist.shape[0])
    density_test = kde_test(x_test)

    plt.figure()
    plt.plot(x_reference, density_reference, color='tab:orange', label="reference")
    plt.fill_between(x_reference, density_reference, color='tab:orange', alpha=0.4)

    plt.plot(x_test, density_test, color='tab:olive', label='test')
    plt.fill_between(x_test, density_test, color='tab:olive', alpha=0.4)
    plt.legend()
    # plt.show()

    w_dist = wasserstein_distance(ref_dist, test_dist)
    w_dists.append(w_dist)

plt.figure()
plt.plot(w_dists, '-o')
plt.xlabel("Latent dims")
plt.ylabel("Wasserstein distance")
plt.show()











synthetic_shapes = np.array([control_points[:, :] + synthetic_shapes[i, :, :] for i in range(synthetic_shapes.shape[0])])
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
    plt.show()