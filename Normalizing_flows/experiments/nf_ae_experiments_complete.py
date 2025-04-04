import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
from tqdm import tqdm
from scipy.stats import gaussian_kde
from scipy.stats import wasserstein_distance
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import randomized_svd
import os
from scipy.stats import qmc


scaler = MinMaxScaler()
########################################################################################################################
###### Functions that will be useful during the run ######
########################################################################################################################
def LatinHypercube(dim_sample, low_bounds, upp_bounds, num_samples):
    """
    Function that is used to sample the parameters from a latin hypercube.
    :param dim_sample: The dimension that we sample
    :param low_bounds: lower bound of the sampling interval
    :param upp_bounds: upper bound of the sampling interval
    :param num_samples: number of desired samples
    :return:
    """
    sampler = qmc.LatinHypercube(d=dim_sample)
    sample = sampler.random(n=num_samples)

    l_bounds = low_bounds
    u_bounds = upp_bounds
    sample_params = qmc.scale(sample, l_bounds, u_bounds)
    return sample_params


def load_cp_momenta(file_name):

    # Get the path to the Downloads folder dynamically
    downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")

    file_path = os.path.join(downloads_folder, file_name)
    data = np.loadtxt(file_path)

    return data


class Compare_shape_dists:
    def __init__(self, reference_group, test_group):

        self.reference_group = reference_group
        self.test_group = test_group

    def quadratic_kernel(self, u, v, c=0.1, d=2, slope=1):
        return (slope * u.T @ v + c) ** d

    def linear_kernel(self, u, v, c=0.1):
        return u.T @ v + c

    def gauss_kernel(self, X1, X2, Sigma=0.2):
        dim1 = X1.shape[1]
        dim2 = X2.shape[1]

        norms1 = np.sum(X1 ** 2, axis=0)
        norms2 = np.sum(X2 ** 2, axis=0)

        mat1 = np.tile(norms1, (dim2, 1)).T
        mat2 = np.tile(norms2, (dim1, 1))

        distmat = mat1 + mat2 - 2 * X1.T @ X2  # full distance matrix
        K = np.exp(-distmat / (2 * Sigma ** 2))

        return K

    def Standardize(self, matrix):

        mean = np.mean(matrix, axis=1)
        std = np.std(matrix, axis=1, ddof=1)
        mat_mean_centered = matrix - mean[:, np.newaxis]

        mat_standardized = mat_mean_centered / std[:, np.newaxis]

        return mat_standardized



    def tSNE_plot_dist(self, reference, test):
        num_generated = reference.shape[1]

        z_all = np.concatenate((reference.T, test.T))
        z_embedded = TSNE(n_components=2, learning_rate='auto',
                                    init='random', perplexity=20).fit_transform(z_all)

        plt.figure()
        plt.scatter(z_embedded[:num_generated, 0], z_embedded[:num_generated, 1], color='green', label='reference')
        plt.scatter(z_embedded[num_generated:, 0], z_embedded[num_generated:, 1], color='orange', label='test')
        plt.legend()
        plt.title('t-SNE visualization of reduced bases')
        plt.show()


    def SVD(self, num_components, snapshot_mat, auto=True):
        """
        Function that calculates the rank-r truncated SVD of a matrix
        :param num_components: Number of singular values to consider. This is the truncation threshold for the
        rank-r truncated SVD
        :param snapshot_mat: The snapshot matrix that will be decomposed
        :param auto: Whether to perform an experiment or not
        :param plot: Whether to plot results or not
        :return:
        """
        print("Initializing SVD for stacked snapshot matrix X1")

        if auto:
            system_energy = 0
            # while system_energy < 0.99:
            U, s, Vh = np.linalg.svd(snapshot_mat)
            U_red = U[:, :num_components]

            s_selected = s[:num_components]

            system_energy = np.sum(s_selected) / np.sum(s)
            # num_components += 1

            print(f"SVD completed with {system_energy * 100} % of the system energy explained")
            print(f"The number of singular values used (truncation threshold) is {num_components}")
            print(f"SVD for stacked snapshot matrix X1 finished")
            # compute the "alpha" projection coefficient of the dataset
            rank_r_SVD_error = np.sum(s[num_components:] ** 2) / np.linalg.norm(snapshot_mat, ord='fro') ** 2
            POD_projection_error = 1 - np.sum(s[:num_components] ** 2) / np.sum(s ** 2)

            # Print the LS relative error of rank-r SVD approximation. This error is related to the fraction of kinetic energy
            # that is missing in the approximation of the snapshot matrix
            print(f"The relative rank-r SVD approximation error is {rank_r_SVD_error * 100}%")
            print(
                f"The relative error of POD projection is {POD_projection_error * 100}%\nThis is "
                f"the cumulative energy not captured by the projection")

            return U, U_red



    def VisualiseDists(self, reference_dists, test_dists, custom_title):

        w_dists = []
        for ii in range(reference_dists.shape[0]):
            w_dist = wasserstein_distance(reference_dists[ii], test_dists[ii])
            w_dists.append(w_dist)

        r_to_visualize = 12
        cols = int(r_to_visualize / 2)

        # Create a figure and a 2x5 grid of subplots with a larger figure size
        fig, axes = plt.subplots(nrows=2, ncols=cols, figsize=(19, 9))

        # Loop over rows and columns to customize each subplot
        nrows, ncols = axes.shape
        plot_num = 1
        for i in range(nrows):
            for j in range(ncols):

                ref_dist = reference_dists[j + i * cols]
                test_dist = test_dists[j + i * cols]

                kde_reference = gaussian_kde(ref_dist)
                x_reference = np.linspace(min(ref_dist), max(ref_dist), ref_dist.shape[0])
                density_reference = kde_reference(x_reference)

                kde_test = gaussian_kde(test_dist)
                x_test = np.linspace(min(test_dist), max(test_dist), test_dist.shape[0])
                density_test = kde_test(x_test)

                ax = axes[i, j]
                ax.plot(x_reference, density_reference, color='tab:orange', label="reference")
                ax.fill_between(x_reference, density_reference, color='tab:orange', alpha=0.4)

                ax.plot(x_test, density_test, color='tab:olive', label='test')
                ax.fill_between(x_test, density_test, color='tab:olive', alpha=0.4)

                ax.set_title(f'Reduced basis {plot_num}')
                plot_num += 1

                # Only set the x-axis label and tick labels for the last row
                if i == nrows - 1:
                    ax.set_xlabel('')
                else:
                    ax.set_xlabel('')

                # Only set the y-axis label and tick labels for the first column
                ax.set_ylabel('')
                ax.legend()



        plt.tight_layout()
        plt.title(custom_title)
        plt.show()


        return w_dists


    def PCA_Experiment(self, custom_title, num_components, standardize=True, rand_svd=False):

        ### Perform the PCA on the reference group
        X_mean_ref = np.mean(self.reference_group, axis=1)
        X_ref = self.reference_group - X_mean_ref[:, np.newaxis]

        X_mean_test = np.mean(self.test_group, axis=1)
        X_test = self.test_group - X_mean_test[:, np.newaxis]

        if not rand_svd:
            _, U_red_reference = self.SVD(num_components=num_components, snapshot_mat=X_ref)
        else:
            U_red_reference, _, _ = randomized_svd(X_ref, n_components=num_components, random_state=0)

        X_red_reference = U_red_reference.T @ X_ref
        X_red_test = U_red_reference.T @ X_test

        if standardize:
            X_red_reference_dist = self.Standardize(X_red_reference)
            X_red_test_dist = self.Standardize(X_red_test)
            w_dists = self.VisualiseDists(reference_dists=X_red_reference_dist, test_dists=X_red_test_dist,
                                          custom_title=custom_title)
        else:

            w_dists = self.VisualiseDists(reference_dists=X_red_reference, test_dists=X_red_test,
                                          custom_title=custom_title)

        ### Visualize the t_sne plot of the NOT STANDARDIZED distribution
        self.tSNE_plot_dist(reference=X_red_reference, test=X_red_test)

        return w_dists



class Compare_z_synthetic:
    def __init__(self, z_real, z_synthetic):

        self.z_real = z_real
        self.z_synthetic = z_synthetic


    def t_sne(self, female_indices, male_indices):

        num_real = self.z_real.shape[0]

        z_all = np.concatenate((self.z_real, self.z_synthetic))
        z_tsne = TSNE(n_components=2, learning_rate='auto',
                      init='random', perplexity=20).fit_transform(z_all)

        z_tsne_real = z_tsne[:num_real, :]
        z_tsne_synthetic = z_tsne[num_real:, :]

        plt.figure(figsize=(10, 7))
        plt.scatter(z_tsne_real[female_indices, 0], z_tsne_real[female_indices, 1],
                    color='red', marker='o', label='female real', alpha=0.7)
        plt.scatter(z_tsne_real[male_indices, 0], z_tsne_real[male_indices, 1],
                    color='blue', marker='o', label='male real', alpha=0.7)

        plt.scatter(z_tsne_synthetic[female_indices, 0], z_tsne_synthetic[female_indices, 1],
                    color='red', marker='+', label='female gen')
        plt.scatter(z_tsne_synthetic[male_indices, 0], z_tsne_synthetic[male_indices, 1],
                    color='blue', marker='+', label='male gen')
        plt.legend()
        plt.title('t-SNE: Posterior distribution')
        plt.show()


    def Wasserstein_dist(self):
        w_dists = []
        for i in range(self.z_real.shape[1]):
            ref_dist = self.z_real[:, i]
            test_dist = self.z_synthetic[:, i]

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
        plt.xlabel("Latent dimensions")
        plt.ylabel("Wasserstein distance")
        plt.show()

        return w_dists

    def flow_transform(self, z_outs):

        z_tsne_plots = []
        bar = tqdm(total=len(z_outs), desc = 't-SNE calculation...', leave=True)
        z_tsne_plots = []
        for z in z_outs:
            z = z.detach().cpu().numpy()
            z_tsne = TSNE(n_components=2, learning_rate='auto',
                          init='random', perplexity=20).fit_transform(z)
            z_tsne_plots.append(z_tsne)
            bar.update()
        bar.close()

        # Set up the initial plot
        fig, ax = plt.subplots()
        scatt = ax.scatter(z_tsne_plots[0][:, 0], z_tsne_plots[0][:, 1])
        ax.set_xlabel('z_1')
        ax.set_ylabel('z_2')

        # Define the update function for the animation
        def update(frame):
            # Update scatter plot data
            scatt.set_offsets(z_tsne_plots[frame])
            # Update title with the current step
            ax.set_title(f'Flow transformation - Step {frame}')
            return scatt,

        # Create the animation
        ani = FuncAnimation(fig, update, frames=len(z_tsne_plots), interval=500, blit=False)

        plt.show()


    def visualize_synthetic_momenta(self, synthetic_shapes, control_points):

        synthetic_shapes = np.array(
            [control_points[:, :] + synthetic_shapes[i, :, :] for i in range(synthetic_shapes.shape[0])])
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
            ax.set_title(f'Synthetic shape {i}')
            plt.show()


    def visualize_prior(self, z_prior, female_indices, male_indices, plot_1d=False):
        z_tsne_prior = TSNE(n_components=2, learning_rate='auto',
                      init='random', perplexity=20).fit_transform(z_prior)



        plt.figure()
        plt.scatter(z_tsne_prior[female_indices, 0], z_tsne_prior[female_indices, 1],
                    color='red', label='female gen')
        plt.scatter(z_tsne_prior[male_indices, 0], z_tsne_prior[male_indices, 1],
                    color='blue', label='male gen')
        plt.legend()
        plt.title('t-SNE: Learned prior distribution')
        plt.show()


        if plot_1d:
            latent_dimension = z_prior.shape[1]
            z_prior_female = z_prior[female_indices]
            z_prior_male = z_prior[male_indices]
            for i in range(latent_dimension):
                kde = gaussian_kde(z_prior[:, i])
                x = np.linspace(min(z_prior[:, i]), max(z_prior[:, i]), z_prior[:, i].shape[0])
                density = kde(x)

                plt.figure()
                plt.hist(z_prior[:, i], bins=25, color='b', density=True, alpha=0.3, label="Histogram")
                plt.plot(x, density, label="KDE", color="red")
                plt.scatter(z_prior_female[:, i], np.abs(np.random.randn(z_prior_female[:, i].size)) / 40, marker='x',
                            color='red', label="female")
                plt.scatter(z_prior_male[:, i], np.abs(np.random.randn(z_prior_male[:, i].size)) / 40, marker='o',
                            color='green', alpha=0.2, label="male")
                plt.title(fr"Learned prior dist: z_{i}")
                plt.legend()
                plt.show()




########################################################################################################################
###### Configurations of the run ######
########################################################################################################################


frac_train = 0.5
frac_valid = 0.45

NumTrainSamples = int(456 * frac_train)
NumValidSamples = int(456 * frac_valid)
NumTestSamples = 456 - NumTrainSamples - NumValidSamples


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


########################################################################################################################
###### Data loading & pre processing ######
########################################################################################################################


file_name_cp = r"cardiac_atlas_kostas\output\DeterministicAtlas__EstimatedParameters__ControlPoints.txt"
file_name_momenta = r"cardiac_atlas_kostas\output\DeterministicAtlas__EstimatedParameters__Momenta.txt"
file_path_confounders = r'C:\Users\kkevopoulos\Downloads\cardiac_function_mesh_complete_final.xlsx'

cp = load_cp_momenta(file_name_cp)
momenta = load_cp_momenta(file_name_momenta)
momenta = np.delete(momenta, 0, axis=0)
momenta = momenta.reshape((456, 2730, 3))
momenta = momenta.reshape((456, 3, 13, 14, 15))


x_confounders = pd.read_excel(file_path_confounders)
x_confounders.drop(['BSA', 'Year of birth', 'Height', 'Weight', 'Diastolic BP mean reading',
                 'Systolic BP mean reading', 'mesh_file', 'MAP'], axis=1, inplace=True)
x_confounders = pd.get_dummies(x_confounders, columns=['Sex'])
x_confounders['Sex_Female'] = x_confounders['Sex_Female'].replace({True: 1, False: 0})
x_confounders['Sex_Male'] = x_confounders['Sex_Male'].replace({True: 1, False: 0})
x_confounders = x_confounders.to_numpy()
x_confounders = x_confounders[:, 1:]
x_confounders = scaler.fit_transform(x_confounders)

male_indices = np.where(x_confounders[:, 3].flatten() == 1)[0]
female_indices = np.where(x_confounders[:, 2].flatten() == 1)[0]



########################################################################################################################
###### Model loading & experiments ######
########################################################################################################################
encoder = torch.load("../data_models_saved/encoder.pth")
decoder = torch.load("../data_models_saved/decoder.pth")
norm_flow = torch.load("../data_models_saved/flow_model.pth")

encoder.eval()
decoder.eval()
norm_flow.eval()
########################################################################################################################
###### EXPERIMENT CLASS 1: COMPARE LATENT DISTRIBUTIONS z_synthetic AND z_real
########################################################################################################################

### Do this for all the 456 patient samples
real_z = encoder(torch.tensor(momenta, dtype=torch.float32).to(device))
synthetic_z, outs = norm_flow.reverse(x_conf=torch.tensor(x_confounders, dtype=torch.float32).to(device), animate=True)

real_z =real_z.detach().cpu().numpy()
synthetic_z = synthetic_z.detach().cpu().numpy()

# compare_z_latent = Compare_z_synthetic(z_real=real_z, z_synthetic=synthetic_z)
#
# ### Measure the Wasserstein distance between real_z and synthetic_z distributions, and also visualize the distributions
# w_dists = compare_z_latent.Wasserstein_dist()
#
#
# ### Visualize posterior distributions of real_z and synthetic_z
# compare_z_latent.t_sne(female_indices=female_indices, male_indices=male_indices)
#
# # ### Visualize the (learned) prior distribution of z_synthetic
# compare_z_latent.visualize_prior(z_prior=outs[0].detach().cpu().numpy(), female_indices=female_indices, male_indices=male_indices)
#
#
# ### Visualize synthtic momenta shapes --> to make sure they are realistic
# synthetic_shapes = decoder(torch.tensor(synthetic_z, dtype=torch.float32).to(device))
# synthetic_shapes = synthetic_shapes.detach().cpu().numpy()
# synthetic_shapes = synthetic_shapes.reshape(synthetic_shapes.shape[0], 2730, 3)
#
# compare_z_latent.visualize_synthetic_momenta(synthetic_shapes=synthetic_shapes, control_points=cp)
#
#
# ### Visualize the flow transformation
# compare_z_latent.flow_transform(z_outs=outs)



########################################################################################################################
###### EXPERIMENT CLASS 2: COMPARE MOMENTA SHAPES momenta_real V momenta_synthetic
########################################################################################################################
latent_dimension = synthetic_z.shape[1]

########################################################################################################################
###### EXPERIMENT 1: TRAINING DATA
###### compare generated vs real momenta, regardless of the metadata
###### Just want to see if the generated and real shapes follow the same (similar) distributions

###### For each confounder in the training dataset, generate a synthetic sample. Then compare the two distributions
########################################################################################################################
X_train_momenta = momenta[:NumTrainSamples, :, :, :, :]
X_valid_momenta = momenta[NumTrainSamples:NumTrainSamples + NumValidSamples, :, :, :, :]
X_test_momenta = momenta[NumTrainSamples + NumValidSamples:, :, :, :, :]

X_train_conf = torch.tensor(x_confounders[:NumTrainSamples], dtype=torch.float32).to(device)
X_valid_conf = torch.tensor(x_confounders[NumTrainSamples:NumTrainSamples + NumValidSamples], dtype=torch.float32).to(device)
X_test_conf = torch.tensor(x_confounders[NumTrainSamples + NumValidSamples:], dtype=torch.float32).to(device)

### Generate synthetic z with norm flow model --> and then shapes with the AE
synthetic_z_train = norm_flow.reverse(X_train_conf, animate=False)
synthetic_shapes_train = decoder(synthetic_z_train)

synthetic_shapes_train = synthetic_shapes_train.detach().cpu().numpy()
synthetic_shapes_train = synthetic_shapes_train.reshape(NumTrainSamples, 2730, 3)
synthetic_group = synthetic_shapes_train.transpose(1, 2, 0).reshape(2730 * 3, NumTrainSamples)

reference_group = X_train_momenta.reshape(NumTrainSamples, 2730, 3)
reference_group = reference_group.transpose(1, 2, 0).reshape(2730 * 3, NumTrainSamples)

diff = Compare_shape_dists(reference_group=reference_group, test_group=synthetic_group)

w_diff = diff.PCA_Experiment(custom_title='experiment training', num_components=600, rand_svd=True, standardize=False)

plt.figure()
plt.plot(w_diff, color='black', label='Real vs Generated')
plt.legend()
plt.xlabel('Reduced basis')
plt.ylabel('Wasserstein distance')
plt.show()



########################################################################################################################
###### Experiment2: TEST DATA
###### For each metadata vector in the validation + test dataset, generate "N" number of probable shapes
###### So I'll have "N_real" reference shapes, and "N * N_real" generated shapes
###### Compare these two distributions
########################################################################################################################
NumGen = 1

X_test_momenta_exp2 = np.concatenate((X_valid_momenta, X_test_momenta), axis=0)
X_test_exp_conf = torch.cat((X_valid_conf, X_test_conf), dim=0)

synthetic_shapes_exp2 = np.zeros((2730*3, NumGen*(NumValidSamples+NumTestSamples)))


for i in range(NumValidSamples+NumTestSamples):
    for j in range(NumGen):


        synthetic_z_exp2 = norm_flow.reverse(X_test_exp_conf[i].reshape(-1, 1).T)
        synthetic_shape_exp2 = decoder(synthetic_z_exp2)


        gen_shape = synthetic_shape_exp2.reshape(1, 2730, 3)
        gen_shape = gen_shape.detach().cpu().numpy()
        gen_shape = gen_shape.transpose(1, 2, 0).reshape(2730 * 3, )

        synthetic_shapes_exp2[:, NumGen * i + j] = gen_shape

reference_shapes_exp2 = X_test_momenta_exp2.reshape(NumValidSamples+NumTestSamples, 2730, 3).transpose(1, 2, 0).reshape(2730 * 3, (NumValidSamples+NumTestSamples))


diff = Compare_shape_dists(reference_group=reference_shapes_exp2, test_group=synthetic_shapes_exp2)
w_diff = diff.PCA_Experiment(custom_title='experiment test', num_components=1000, rand_svd=True, standardize=False)

plt.figure()
plt.semilogy(w_diff, color='black', label='Real vs Generated')
plt.legend()
plt.xlabel('Reduced basis')
plt.ylabel('Wasserstein distance')
plt.show()



########################################################################################################################
###### Experiment 3: compare generated vs real momenta, when only the sex is different
###### For a specific sex --> sample "N" number of (age, BMI)

###### Compare the generated shapes with real shapes from the same sex. Have different sub-groups for that specific sex
###### Such that I can compare the performance of (test vs ref) with (ref vs group) --> groups are also reference data
########################################################################################################################
num_generated = 90
generated_sex_1 = 1 ### female
generated_sex_1 = np.tile(generated_sex_1, num_generated).reshape(-1, 1)
generated_sex_2 = np.tile(0, num_generated).reshape(-1, 1)
generated_sex = np.concatenate((generated_sex_1, generated_sex_2), axis=1)

### Inverse the scaling of the confounders, just for the sampling
x_confounders = scaler.inverse_transform(x_confounders)
generated_bmi_female = np.random.uniform(low=np.min(x_confounders[female_indices][:, 0]), high=np.max(x_confounders[female_indices][:, 0]),
                                  size=num_generated).reshape(-1, 1)
generated_bmi_female = scaler.fit_transform(generated_bmi_female)

generated_age_female = np.random.randint(low=np.min(x_confounders[female_indices][:, 1]), high=np.max(x_confounders[female_indices][:, 1]),
                                  size=num_generated).reshape(-1, 1)
generated_age_female = scaler.fit_transform(generated_age_female)

generated_conf_samples = np.concatenate((generated_bmi_female, generated_age_female, generated_sex), axis=1)
generated_conf_samples = torch.tensor(generated_conf_samples, dtype=torch.float32).to(device)




generated_bmi_male = np.random.uniform(low=np.min(x_confounders[male_indices][:, 0]), high=np.max(x_confounders[male_indices][:, 0]),
                                  size=num_generated).reshape(-1, 1)
generated_bmi_male = scaler.fit_transform(generated_bmi_male)

generated_age_male = np.random.randint(low=np.min(x_confounders[male_indices][:, 1]), high=np.max(x_confounders[male_indices][:, 1]),
                                  size=num_generated).reshape(-1, 1)
generated_age_male = scaler.fit_transform(generated_age_male)

sex_male_1 = 0
sex_male_2 = 1

sex_male_1 = np.tile(sex_male_1, num_generated).reshape(-1, 1)
sex_male_2 = np.tile(sex_male_2, num_generated).reshape(-1, 1)
generated_sex_male = np.concatenate((sex_male_1, sex_male_2), axis=1)

generated_conf_samples_male = np.concatenate((generated_bmi_male, generated_age_male, generated_sex_male), axis=1)

z_synth_female = norm_flow.reverse(generated_conf_samples)
gen_shapes_female = decoder(z_synth_female)

gen_shapes_female = gen_shapes_female.detach().cpu().numpy()
gen_shapes_female = gen_shapes_female.reshape(num_generated, 2730, 3)
gen_shapes_female = gen_shapes_female.transpose(1, 2, 0).reshape(2730 * 3, num_generated)


male_group1_ind = male_indices[:num_generated]
male_group2_ind = male_indices[num_generated:2*num_generated]
female_ind = female_indices[num_generated:2*num_generated]


male_momenta_group1 = momenta[male_group1_ind, :, :, :, :]
male_momenta_group2 = momenta[male_group2_ind, :, :, :, :]
female_momenta = momenta[female_ind, :, :, :, :]



male_momenta_group1 = male_momenta_group1.reshape(num_generated, 2730, 3)
male_momenta_group1 = male_momenta_group1.transpose(1, 2, 0).reshape(2730 * 3, num_generated)

male_momenta_group2 = male_momenta_group2.reshape(num_generated, 2730, 3)
male_momenta_group2 = male_momenta_group2.transpose(1, 2, 0).reshape(2730 * 3, num_generated)

female_momenta = female_momenta.reshape(num_generated, 2730, 3)
female_momenta = female_momenta.transpose(1, 2, 0).reshape(2730 * 3, num_generated)


male_female_comparison = Compare_shape_dists(reference_group=male_momenta_group1, test_group=gen_shapes_female)
male1_female_comparison = Compare_shape_dists(reference_group=male_momenta_group2, test_group=gen_shapes_female)
female_female_comparison = Compare_shape_dists(reference_group=female_momenta, test_group=gen_shapes_female)


w_dists_male_female = male_female_comparison.PCA_Experiment(custom_title='Male real Vs Female generated',
                                                            num_components=90, standardize=False)
w_dists_male1_female = male1_female_comparison.PCA_Experiment(custom_title='Male real alt Vs Female generated',
                                                              num_components=90, standardize=False)
w_dists_female_female = female_female_comparison.PCA_Experiment('Female real Vs Female generated',
                                                                num_components=90, standardize=False)

plt.figure()
plt.plot(w_dists_male_female, color='black', label='Male V Female')
plt.plot(w_dists_male1_female, color='green', label='Male1 V Female')
plt.plot(w_dists_female_female, color='orange', label='Female V Female')
plt.legend()
plt.xlabel('Reduced basis')
plt.ylabel('Wasserstein distance')
plt.show()



###### Compare generated female V generated male shapes
z_synthetic_male_exp3 = norm_flow.reverse(torch.tensor(generated_conf_samples_male, dtype=torch.float32).to(device))
gen_shapes_male = decoder(z_synthetic_male_exp3)

gen_shapes_male = gen_shapes_male.detach().cpu().numpy()
gen_shapes_male = gen_shapes_male.reshape(num_generated, 2730, 3)
gen_shapes_male = gen_shapes_male.transpose(1, 2, 0).reshape(2730 * 3, num_generated)

gen_female_gen_male_comp = Compare_shape_dists(reference_group=gen_shapes_female,
                                                    test_group=gen_shapes_male)

### 2nd Experiment --> Common PCA
w_dists_gfemale_gmale = gen_female_gen_male_comp.PCA_Experiment(custom_title='female gen Vs male gen',
                                                                num_components=90, standardize=False)

plt.figure()
plt.plot(w_dists_gfemale_gmale, color='black', label='female gen V male gen')
plt.legend()
plt.xlabel('Reduced basis')
plt.ylabel('Wasserstein distance')
plt.show()


likelihoods_female_female = norm_flow(z_synth_female, x_conf=generated_conf_samples, compute_likelihoods=True)
likelihoods_female_male = norm_flow(z_synth_female, x_conf=torch.tensor(generated_conf_samples_male, dtype=torch.float32).to(device), compute_likelihoods=True)

plt.hist(-likelihoods_female_female.detach().cpu().numpy(), bins=20, edgecolor='black', alpha=0.5, label='fem|fem')
plt.hist(-likelihoods_female_male.detach().cpu().numpy(), bins=20,  edgecolor='black', alpha=0.5, label='fem|male')
plt.legend()
plt.show()

print(f'alpha={likelihoods_female_female.detach().cpu().numpy().mean() / likelihoods_female_male.detach().cpu().numpy().mean() }')



########################################################################################################################
###### Experiment4: compare generated vs synthetic momenta, when (sex, age, BMI) are all varied.
###### In the 3D parameter space, we consider 3 smaller sub-spaces. We call these "sample islands".
###### We consider Latin HyperCube Sampling to sample metadata vectors from these sub-islands
###### Compare the generated shapes with real shapes in these sub-islands.
###### Also, compare the generated shapes of sample_island1, with the real shapes of sample_island2
########################################################################################################################
### First sample island is: (sex, age, BMI) = (0, [45, 57], [21,27])
age_min1 = 45.
age_max1 = 57.
bmi_min1 = 21.
bmi_max1 = 27.

mask1 = (
    (x_confounders[:, 0] >= bmi_min1) & (x_confounders[:, 0] <= bmi_max1) &
    (x_confounders[:, 1] >= age_min1) & (x_confounders[:, 1] <= age_max1) &
    (x_confounders[:, 2] == 0) &
    (x_confounders[:, 3] == 1)

)

x_conf_isl1 = x_confounders[mask1]
momenta_isl1 = momenta[mask1, :, :, :, :]


### Second sample island is: (sex, age, BMI) = (1, [65, 75], [24, 30])
age_min2 = 65.
age_max2 = 75.
bmi_min2 = 24.
bmi_max2 = 30.

mask2 = (
    (x_confounders[:, 0] >= bmi_min2) & (x_confounders[:, 0] <= bmi_max2) &
    (x_confounders[:, 1] >= age_min2) & (x_confounders[:, 1] <= age_max2) &
    (x_confounders[:, 2] == 1) &
    (x_confounders[:, 3] == 0)
)

x_conf_isl2 = x_confounders[mask2]
momenta_isl2 = momenta[mask2, :, :, :, :]


### Third sample island is: (sex, age, BMI) = (1, [45, 55], [18, 25])
age_min3 = 45.
age_max3 = 55.
bmi_min3 = 18.
bmi_max3 = 25.

mask3 = (
    (x_confounders[:, 0] >= bmi_min3) & (x_confounders[:, 0] <= bmi_max3) &
    (x_confounders[:, 1] >= age_min3) & (x_confounders[:, 1] <= age_max3) &
    (x_confounders[:, 2] == 1) &
    (x_confounders[:, 3] == 0)
)

x_conf_isl3 = x_confounders[mask3]
momenta_isl3 = momenta[mask3, :, :, :, :]


### Now for each sample island, generate samples of confounders. These samples will then be used to generate shapes
conf_to_generate_isl1 = LatinHypercube(dim_sample=2, low_bounds=[bmi_min1, age_min1],
                                       upp_bounds=[bmi_max1, age_max1], num_samples=num_generated)
conf_to_generate_isl1 = scaler.fit_transform(conf_to_generate_isl1)
sex_isl1 = np.tile(0, num_generated).reshape(-1, 1)
sex1_isl1 = np.tile(1, num_generated).reshape(-1, 1)
sex_island1 = np.concatenate((sex_isl1, sex1_isl1), axis=1)

conf_to_generate_isl1 = np.concatenate((conf_to_generate_isl1, sex_island1), axis=1)




conf_to_generate_isl2 = LatinHypercube(dim_sample=2, low_bounds=[bmi_min2, age_min2],
                                       upp_bounds=[bmi_max2, age_max2], num_samples=num_generated)
conf_to_generate_isl2 = scaler.fit_transform(conf_to_generate_isl2)
sex_isl2 = np.tile(1, num_generated).reshape(-1, 1)
sex1_isl2 = np.tile(0, num_generated).reshape(-1, 1)
sex_island2 = np.concatenate((sex_isl2, sex1_isl2), axis=1)

conf_to_generate_isl2 = np.concatenate((conf_to_generate_isl2, sex_island2), axis=1)





conf_to_generate_isl3 = LatinHypercube(dim_sample=2, low_bounds=[bmi_min3, age_min3],
                                       upp_bounds=[bmi_max3, age_max3], num_samples=num_generated)
conf_to_generate_isl3 = scaler.fit_transform(conf_to_generate_isl3)
sex_isl3 = np.tile(1, num_generated).reshape(-1, 1)
sex1_isl3 = np.tile(0, num_generated).reshape(-1, 1)
sex_island3 = np.concatenate((sex_isl3, sex1_isl3), axis=1)

conf_to_generate_isl3 = np.concatenate((conf_to_generate_isl3, sex_island3), axis=1)



### For each sample island, given these sample confounders generate shapes
confs_isl_to_generate = [conf_to_generate_isl1, conf_to_generate_isl2, conf_to_generate_isl3]
momenta_islands = [momenta_isl1, momenta_isl2, momenta_isl3]
generated_shapes_islands = []

for conf_island in confs_isl_to_generate:

    conf_island = torch.tensor(conf_island, dtype=torch.float32).to(device)

    z_synthetic_island = norm_flow.reverse(conf_island)
    gen_shapes_island = decoder(z_synthetic_island)

    generated_shapes_islands.append(gen_shapes_island)



for i, conf_island in enumerate(confs_isl_to_generate):
    reference_group = momenta_islands[i].reshape(momenta_islands[i].shape[0], 2730, 3)
    reference_group = reference_group.transpose(1, 2, 0).reshape(2730 * 3, momenta_islands[i].shape[0])

    test_group = generated_shapes_islands[i].reshape(generated_shapes_islands[i].shape[0], 2730, 3)
    test_group = test_group.detach().cpu().numpy()
    test_group = test_group.transpose(1, 2, 0).reshape(2730 * 3, num_generated)

    main_comparison = Compare_shape_dists(reference_group=reference_group, test_group=test_group)

    ### 2nd Experiment --> Common PCA
    w_dists_main = main_comparison.PCA_Experiment(custom_title='Real Vs Generated', num_components=90, standardize=False)

    plt.figure()
    plt.plot(w_dists_main, color='black', label='Real Vs Generated')
    plt.xlabel('Reduced basis')
    plt.ylabel('Wasserstein distance')
    plt.legend()
    plt.show()






### Compare sample island 1 (generated) with sample island 3 (real)
reference_group = momenta_isl3.reshape(momenta_isl3.shape[0], 2730, 3)
reference_group = reference_group.transpose(1, 2, 0).reshape(2730 * 3, momenta_isl3.shape[0])

test_group = generated_shapes_islands[0].reshape(generated_shapes_islands[0].shape[0], 2730, 3)
test_group = test_group.detach().cpu().numpy()
test_group = test_group.transpose(1, 2, 0).reshape(2730 * 3, num_generated)

main_comparison = Compare_shape_dists(reference_group=reference_group, test_group=test_group)

### 2nd Experiment --> Common PCA
w_dists_main = main_comparison.PCA_Experiment(custom_title='Real female s2 Vs Gen male',
                                              num_components=90, standardize=False)

plt.figure()
plt.plot(w_dists_main, '-o', color='black', label='Real female s2 Vs Gen male')
plt.xlabel('Reduced basis')
plt.ylabel('Wasserstein distance')
plt.legend()
plt.show()





### Compare sample island 2 (generated) with sample island 3 (real)
reference_group = momenta_isl3.reshape(momenta_isl3.shape[0], 2730, 3)
reference_group = reference_group.transpose(1, 2, 0).reshape(2730 * 3, momenta_isl3.shape[0])

test_group = generated_shapes_islands[1].reshape(generated_shapes_islands[1].shape[0], 2730, 3)
test_group = test_group.detach().cpu().numpy()
test_group = test_group.transpose(1, 2, 0).reshape(2730 * 3, num_generated)

main_comparison = Compare_shape_dists(reference_group=reference_group, test_group=test_group)

### 2nd Experiment --> Common PCA
w_dists_main = main_comparison.PCA_Experiment(custom_title='Real female s2 Vs Gen female s3',
                                              num_components=90, standardize=False)

plt.figure()
plt.plot(w_dists_main, '-o', color='black', label='Real female s2 Vs Gen female s3')
plt.xlabel('Reduced basis')
plt.ylabel('Wasserstein distance')
plt.legend()
plt.show()
