import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from matplotlib import rcParams, font_manager
import numpy as np
from scipy.stats import gaussian_kde


### This file visualizes the conditional base density of CAN-FLOW
### This file is used to generate Figure B.14 of the manuscript


seed = 80
np.random.seed(seed)


torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def t_sne(z_base, female_indices, male_indices):
    """
        Compute and visualize a 2D t-SNE embedding of latent representations, colored by sex.

        This function applies t-SNE to a set of latent representations and visualizes
        the resulting two-dimensional embedding. Female and male samples are shown as
        scatter points with separate colors. In addition, two-dimensional KDE contours
        are plotted for each group to highlight the density structure of the embedded
        latent space.


        Parameters
        ----------
        z_base : numpy.ndarray
            Array containing the latent vectors to be embedded with t-SNE.

            Shape:

            `(n_samples, n_latent_dimensions)`

            where `n_samples` is the number of samples and `n_latent_dimensions`
            is the dimensionality of the latent representation.

        female_indices : array-like
            index array selecting the female samples in `z_base`.

        male_indices : array-like
            index array selecting the male samples in `z_base`.
        """

    folder = "../figures_experiments/t_sne_approximations/"
    os.makedirs(folder, exist_ok=True)

    z_tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=15).fit_transform(z_base)


    ### Visualize wrt gender
    xmin, xmax = (min(z_tsne[female_indices, 0].min(), z_tsne[male_indices, 0].min()) - 10,
                  max(z_tsne[female_indices, 0].max(), z_tsne[male_indices, 0].max()) + 10)

    ymin, ymax = (min(z_tsne[female_indices, 1].min(), z_tsne[male_indices, 1].min()) - 10,
                  max(z_tsne[female_indices, 1].max(), z_tsne[male_indices, 1].max()) + 10)

    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 500),
                         np.linspace(ymin, ymax, 500))

    ### Figure 1: real data
    plt.figure(figsize=(10, 7))
    plt.scatter(z_tsne[female_indices, 0], z_tsne[female_indices, 1],
                color='purple', marker='o', label='female real', alpha=0.15)
    plt.scatter(z_tsne[male_indices, 0], z_tsne[male_indices, 1],
                color='teal', marker='o', label='male real', alpha=0.15)
    ### KDE for females
    f_kde = gaussian_kde(np.vstack([z_tsne[female_indices, 0], z_tsne[female_indices, 1]]))
    f_z = f_kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    plt.contour(xx, yy, f_z, colors='purple', levels=7, linewidths=1.6)

    ### KDE for males
    m_kde = gaussian_kde(np.vstack([z_tsne[male_indices, 0], z_tsne[male_indices, 1]]))
    m_z = m_kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    plt.contour(xx, yy, m_z, colors='teal', levels=7, linewidths=1.6)
    plt.show()




### Figure formatting
params = {
   'axes.labelsize': 20,
   'font.size': 20,
   'legend.fontsize': 20,
   'xtick.labelsize': 20,
   'ytick.labelsize': 20,
   'text.usetex': False,
   'axes.linewidth': 2.2,
   'xtick.major.width': 2.2,
   'ytick.major.width': 2.2,
   'xtick.major.size': 2.2,
   'ytick.major.size': 2.2,
}
plt.rcParams.update(params)
font_path = r'C:\Users\kkevopoulos\AppData\Local\Microsoft\Windows\Fonts\SourceSansPro-Regular.otf'
font_prop = font_manager.FontProperties(fname=font_path)
rcParams['font.family'] = font_prop.get_name()


device = 'cuda'

scaler = MinMaxScaler()


x_confounders = pd.read_excel(r"/home/kevopou1/metadata_final.xlsx")
x_confounders.drop(['Participant ID', 'Height', 'Weight', 'Diastolic BP',
                    'Systolic BP', 'Unnamed: 8', 'Unnamed: 9', 'subject_id'], axis=1, inplace=True)
x_confounders = pd.get_dummies(x_confounders, columns=['Sex'])
x_confounders['Sex_Female'] = x_confounders['Sex_Female'].replace({True: 1, False: 0})
x_confounders['Sex_Male'] = x_confounders['Sex_Male'].replace({True: 1, False: 0})
x_confounders = x_confounders[['BMI', 'Age', 'Sex_Female', 'Sex_Male']]
x_confounders = x_confounders.to_numpy()

### Delete outliers and participants that withdrew from the study
x_confounders = np.delete(x_confounders, [1746, 1831], axis=0)

outliers = np.load('../utils/outliers_indices.npy')
mask = np.ones(x_confounders.shape[0], dtype=bool)
mask[outliers] = False

x_confounders = x_confounders[mask]

male_indices = np.where(x_confounders[:, 2].flatten() == 0)[0]
female_indices = np.where(x_confounders[:, 2].flatten() == 1)[0]

NumAll = 2208
frac_train = 0.7
frac_valid = 0.15

NumTrainSamples = int(frac_train * NumAll)
NumValidSamples = int(frac_valid * NumAll)

X_train_confounders = scaler.fit_transform(x_confounders[:NumTrainSamples, :])
X_valid_confounders = scaler.transform(x_confounders[NumTrainSamples:NumTrainSamples+NumValidSamples, :])
X_test_confounders = scaler.transform(x_confounders[NumTrainSamples+NumValidSamples:])

X_train_confounders = torch.tensor(X_train_confounders, dtype=torch.float32).to(device)
X_valid_confounders = torch.tensor(X_valid_confounders, dtype=torch.float32).to(device)
X_test_confounders = torch.tensor(X_test_confounders, dtype=torch.float32).to(device)


### Load CAN-FLOW to compute the learned base distribution
normalizing_flow = torch.load("../data_models_saved/models/cnf_model.pth", weights_only=False)
normalizing_flow.eval()

_, outs_train_all = normalizing_flow.reverse(X_train_confounders)
_, outs_valid_all = normalizing_flow.reverse(X_valid_confounders)
_, outs_test_all = normalizing_flow.reverse(X_test_confounders)

### choose the first element of the output list -- corresponds to the base density
base_train = outs_train_all[0].detach().cpu().numpy()
base_valid = outs_valid_all[0].detach().cpu().numpy()
base_test = outs_test_all[0].detach().cpu().numpy()

base_all = np.concatenate((base_train, base_valid, base_test), axis=0)

t_sne(z_base=base_all, female_indices=female_indices, male_indices=male_indices)
