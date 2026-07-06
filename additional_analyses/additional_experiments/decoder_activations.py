import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
import os
import torch.nn.functional as F
from scipy.stats import gaussian_kde


os.makedirs('../data_results/activations_metrics', exist_ok=True)


seed = 42
np.random.seed(seed)


torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def effective_rank_linear(a):
    # a: (B, D)
    a = a - a.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(a)

    energy = s ** 2

    p = energy / energy.sum()
    entropy = -(p * torch.log(p + 1e-18)).sum()

    return torch.exp(entropy).item()


def get_linear_activations(decoder, z, x_conf, ae=False):
    decoder.eval()

    activations = {}

    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu()

        return hook

    if not ae:
        h_conf = decoder.conf_embedding.register_forward_hook(get_activation("conf_embedding"))
        h_fc = decoder.fc.register_forward_hook(get_activation("fc"))
    else:
        h_fc = decoder.fc.register_forward_hook(get_activation("fc"))

    with torch.no_grad():
        if not ae:
            _ = decoder(z, x_conf)
        else:
            _ = decoder(z)

    if not ae:
        conf_act = activations["conf_embedding"]
        fc_act = activations["fc"]

        return conf_act, fc_act
    else:
        fc_act = activations["fc"]
        return  fc_act





def effective_rank(activation, eps=1e-18):
    # activation: (N, C, D, H, W)
    N, C, D, H, W = activation.shape
    flat = activation.permute(0, 2, 3, 4, 1).reshape(-1, C)  # (N*D*H*W, C)
    flat = flat - flat.mean(dim=0, keepdim=True)


    S = torch.linalg.svdvals(flat)

    energy = S ** 2
    energy = energy / (energy.sum() + eps)
    entropy = -(energy * (energy + eps).log()).sum()

    return entropy.exp().item()


class DecoderActivationProfiler:
    def __init__(self, decoder):
        self.decoder = decoder
        self.activations = {}
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.decoder.named_modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)

    def _make_hook(self, name):
        def hook(module, inputs, output):
            self.activations[name] = output.detach()
        return hook

    def clear(self):
        self.activations = {}

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def profile(self, z_batch, x_conf_batch, ae):
        self.clear()

        self.decoder.eval()
        with torch.no_grad():

            if ae:
                _ = self.decoder(z_batch)
            else:
                _ = self.decoder(z_batch, x_conf_batch)

        results = {}
        for name, act in self.activations.items():
            C = act.shape[1]
            erank = effective_rank(F.gelu(act))

            results[name] = {
                "shape": tuple(act.shape),
                "num_channels": C,
                "effective_rank": erank,
                "effective_rank_fraction": erank / C,
            }

        return results


def measure_metrics(decoder, z_batch, x_conf_batch=None, ae=True):
    profiler = DecoderActivationProfiler(decoder)

    results = profiler.profile(z_batch, x_conf_batch=x_conf_batch, ae=ae)
    results = dict(list(results.items()))

    profiler.remove()

    return results


params = {
            'axes.labelsize': 31,
            'font.size': 31,
            'legend.fontsize': 31,
            'xtick.labelsize': 31,
            'ytick.labelsize': 31,
            'text.usetex': False,
            'axes.linewidth': 2,
            'xtick.major.width': 2,
            'ytick.major.width': 2,
            'xtick.major.size': 2,
            'ytick.major.size': 2
        }
plt.rcParams.update(params)
font_path = r'C:\Users\kkevopoulos\AppData\Local\Microsoft\Windows\Fonts\SourceSansPro-Regular.otf'
font_prop = font_manager.FontProperties(fname=font_path)
rcParams['font.family'] = font_prop.get_name()



device = 'cuda'


cando_decoder = torch.load('../../data_models_saved/models/ae_decoder.pth')
cando_cnf = torch.load("../../data_models_saved/models/cnf_model.pth")

cvae1_decoder = torch.load('../../data_models_saved/models/cvae_decoder_beta_0.1.pth')
cvae2_decoder = torch.load('../../data_models_saved/models/cvae_decoder_beta_0.01.pth')
cvae3_decoder = torch.load('../../data_models_saved/models/cvae_decoder_beta_0.001.pth')

cando_decoder.eval()
cando_cnf.eval()

cvae1_decoder.eval()
cvae2_decoder.eval()
cvae3_decoder.eval()


### measure metrics firstly for can-do decoder
X_train_z = np.load("../../data_models_saved/data/X_train_z.npy")
X_valid_z = np.load("../../data_models_saved/data/X_valid_z.npy")
X_test_z = np.load("../../data_models_saved/data/X_test_z.npy")


### for cvaes
# x_confounders = pd.read_excel(r"/home/kevopou1/metadata_final.xlsx")
x_confounders = pd.read_excel(r"C:\Users\kkevopoulos\OneDrive - Delft University of Technology\Bureaublad\data_kostas_bivme\metadata_final.xlsx")
x_confounders.drop(['Participant ID', 'Height', 'Weight', 'Diastolic BP',
                    'Systolic BP', 'Unnamed: 8', 'Unnamed: 9', 'subject_id'], axis=1, inplace=True)
x_confounders = pd.get_dummies(x_confounders, columns=['Sex'])
x_confounders['Sex_Female'] = x_confounders['Sex_Female'].replace({True: 1, False: 0})
x_confounders['Sex_Male'] = x_confounders['Sex_Male'].replace({True: 1, False: 0})
x_confounders = x_confounders[['BMI', 'Age', 'Sex_Female', 'Sex_Male']]
x_confounders = x_confounders.to_numpy()

### Delete outliers and participants that withdrew from the study
x_confounders = np.delete(x_confounders, [1746, 1831], axis=0)

outliers = np.load('../../utils/outliers_indices.npy')
mask = np.ones(x_confounders.shape[0], dtype=bool)
mask[outliers] = False

x_confounders = x_confounders[mask]

scaler = MinMaxScaler()

NumTrainSamples = X_train_z.shape[0]
NumValidSamples = X_valid_z.shape[0]
NumTestSamples = X_test_z.shape[0]

X_train_confounders = scaler.fit_transform(x_confounders[:NumTrainSamples, :])


metadata_sampled_female = np.load('../../data_models_saved/data/metadata_sampled_female.npy')
metadata_sampled_male = np.load('../../data_models_saved/data/metadata_sampled_male.npy')

metadata_sampled_all = np.concatenate((metadata_sampled_female, metadata_sampled_male), axis=0)
metadata_sampled_all = scaler.transform(metadata_sampled_all)
metadata_sampled_all = torch.tensor(metadata_sampled_all, dtype=torch.float32, device=device)


### results for can-do
z_cando, _ = cando_cnf.reverse(metadata_sampled_all)
results_can_do = measure_metrics(decoder=cando_decoder, z_batch=z_cando)
fc_activ_cando = get_linear_activations(decoder=cando_decoder, z=z_cando, x_conf=metadata_sampled_all, ae=True)

input_deconvolutions_cando = F.gelu(fc_activ_cando)
input_deconvolutions_cando_std = torch.std(input_deconvolutions_cando, dim=0)



plt.figure()
plt.imshow(input_deconvolutions_cando_std.detach().cpu().numpy().reshape(192, 224), cmap='turbo')
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig('../figures_experiments/activations_metrics/imshow_cando.svg')

erank_cando = effective_rank_linear(input_deconvolutions_cando)
print(f'Effective rank CAN-DO linear layer:', erank_cando)


decoders = [cvae1_decoder, cvae2_decoder, cvae3_decoder]
latent_dim = 44
latent_dist = torch.distributions.MultivariateNormal(torch.zeros(latent_dim), torch.eye(latent_dim))
z_synth_normal = latent_dist.sample((metadata_sampled_all.shape[0],)).to('cuda')

deconv1 = []
deconv2 = []
deconv3 = []

fc_linear_std = []
effective_ranks = []
for i, decoder in enumerate(decoders):

    results = measure_metrics(decoder=decoder, z_batch=z_synth_normal, x_conf_batch=metadata_sampled_all, ae=False)

    deconv1.append(results['deconv1']['effective_rank_fraction'])
    deconv2.append(results['deconv2']['effective_rank_fraction'])
    deconv3.append(results['deconv3']['effective_rank_fraction'])

    conf_activ, fc_activ = get_linear_activations(decoder=decoder, z=z_synth_normal, x_conf=metadata_sampled_all)

    input_deconvolutions_vae = F.gelu(fc_activ)
    input_deconvolutions_vae_std = torch.std(input_deconvolutions_vae, dim=0)

    plt.figure()
    plt.imshow(input_deconvolutions_vae_std.detach().cpu().numpy().reshape(192, 224), cmap='turbo')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f'../figures_experiments/activations_metrics/imshow_cvae_{i+1}.svg')

    fc_linear_std.append(input_deconvolutions_vae_std)

    erank_cvae = effective_rank_linear(input_deconvolutions_vae)
    effective_ranks.append(erank_cvae)
    print(f'Effective rank of cVAE beta={i + 1} linear layer:', erank_cvae)

effective_ranks.append(erank_cando)
fc_linear_std.append(input_deconvolutions_cando_std)

labels = [r'$\beta=10^{-1}$', r'$\beta=10^{-2}$', r'$\beta=10^{-3}$', 'CAN-DO']
colors = ['brown', 'orange', '#CC79A7', '#56B4E9']

all_values = np.concatenate(fc_linear_std[:3])
x_grid = np.linspace(all_values.min(), all_values.max(), 10000)

# fig, ax1 = plt.subplots(figsize=(6.5, 4))
plt.figure()
### cvaes
for arr, label, color in zip(fc_linear_std[:-1], labels[:-1], colors[:-1]):
    kde = gaussian_kde(arr)
    y = kde(x_grid)
    plt.semilogx(x_grid,np.log(y), color=color, linewidth=2, label=label)

plt.show()
# ax1.set_ylabel('cVAEs density')
## can-do
# ax2 = ax1.twinx()
#
# x_grid_cando = np.linspace(fc_linear_std[-1].min(), fc_linear_std[-1].max(), 10000)
#
#
# kde = gaussian_kde(fc_linear_std[-1])
# y = kde(x_grid_cando)
# ax2.semilogx(x_grid, y, linewidth=2, color=colors[-1], label=labels[-1])
# # ax2.set_ylabel('CAN-DO density')
#
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# plt.savefig('../figures_experiments/activations_metrics/distribution_std_linear_decoder.svg')




### Effective rank plot
categories = [r'$\beta=10^{-1}$', r'$\beta=10^{-2}$', r'$\beta=10^{-3}$', 'CAN-DO']

deconv1_erank = [deconv1[i] for i in range(len(decoders))]
deconv1_erank.append(results_can_do['deconv1']['effective_rank_fraction'])

deconv2_erank = [deconv2[i] for i in range(len(decoders))]
deconv2_erank.append(results_can_do['deconv2']['effective_rank_fraction'])

deconv3_erank = [deconv3[i] for i in range(len(decoders))]
deconv3_erank.append(results_can_do['deconv3']['effective_rank_fraction'])


x = np.arange(len(categories))
width = 0.15


fig, ax1 = plt.subplots(figsize=(10, 4))

# LEFT AXIS (effective rank)
ax1.bar(x - 0.5 * width, deconv1_erank, width, label='decoder layer 1', color='limegreen')
ax1.bar(x + 0.5* width,         deconv2_erank, width, label='decoder layer 2', color='olive')
ax1.bar(x + 1.5 * width, deconv3_erank, width, label='decoder layer 3', color='darkgreen')

ax1.set_ylim(0, 0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(categories)


# RIGHT AXIS (new metric)
ax2 = ax1.twinx()

ax2.bar(x - 1.7*width, effective_ranks, width, label='new metric', color='darkblue')
# Color the ticks
ax2.tick_params(axis='y', colors='darkblue')

# Color the spine (the axis line itself)
ax2.spines['right'].set_color('darkblue')
plt.tight_layout()
plt.savefig('../figures_experiments/activations_metrics/decoder_conv_layer_erank_ratio.svg')







