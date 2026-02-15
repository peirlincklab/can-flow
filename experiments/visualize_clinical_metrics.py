import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
from scipy.stats import wasserstein_distance
import seaborn as sns
import os
from scipy.stats import entropy



def kl_div_histogram(x, y):
    px, bins = np.histogram(x, bins=1500, density=False)
    py, _ = np.histogram(y, bins=bins, density=False)

    px = px.astype(float)
    py = py.astype(float)

    # normalize
    px /= px.sum()
    py /= py.sum()

    # avoid zeros
    eps = 1e-12
    px += eps
    py += eps
    px /= px.sum()
    py /= py.sum()

    kl_xy = entropy(px, py)

    return kl_xy




params = {
            'axes.labelsize': 15.4,
            'font.size': 15.4,
            'legend.fontsize': 15.4,
            'xtick.labelsize': 15.4,
            'ytick.labelsize': 15.4,
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



decoder = torch.load("../data_models_saved/models/ae_decoder.pth", weights_only=False, map_location=torch.device('cpu'))


frac_train = 0.7
NumAll = 2274

NumTrainSamples = int(NumAll * frac_train)


df_real = pd.read_pickle('../data_models_saved/data/dataframes_clinical_info/df_real_clinical.pkl')
df_real = df_real.drop(index=1131)

df_nf = pd.read_pickle('../data_models_saved/data/dataframes_clinical_info/df_nf_clinical.pkl')
df_vae2 = pd.read_pickle('../data_models_saved/data/dataframes_clinical_info/df_vae2_clinical.pkl')
df_vae3 = pd.read_pickle('../data_models_saved/data/dataframes_clinical_info/df_vae3_clinical.pkl')

### Compute the WD and KL divergence for clinical metrics, for all models




#### Figure 1: x-axis RV_Vol, y-axis LV_Vol
rv_vol_real = df_real['RV_Vol_mL']
lv_vol_real = df_real['LV_Vol_mL']

rv_vol_nf = df_nf['RV_Vol_mL']
lv_vol_nf= df_nf['LV_Vol_mL']

rv_vol_vae2 = df_vae2['RV_Vol_mL']
lv_vol_vae2 = df_vae2['LV_Vol_mL']

rv_vol_vae3 = df_vae3['RV_Vol_mL']
lv_vol_vae3 = df_vae3['LV_Vol_mL']


fig = plt.figure(figsize=(7, 5))
gs = fig.add_gridspec(2, 2, width_ratios=[4,1], height_ratios=[1,4],
                      wspace=0.05, hspace=0.05)

ax_scatter = fig.add_subplot(gs[1,0])
ax_histx   = fig.add_subplot(gs[0,0], sharex=ax_scatter)
ax_histy   = fig.add_subplot(gs[1,1], sharey=ax_scatter)

ax_histx2 = ax_histx.twinx()
ax_histy2 = ax_histy.twiny()

ax_scatter.scatter(rv_vol_real, lv_vol_real, s=5, color='tab:grey', label='real')
ax_scatter.scatter(rv_vol_nf, lv_vol_nf, s=5, color='aqua', label='CAN-DO')
ax_scatter.scatter(rv_vol_vae2, lv_vol_vae2, s=5, color='tab:orange', label=r'$\beta=10^{-2}$')
ax_scatter.scatter(rv_vol_vae3, lv_vol_vae3, s=5, color='tab:red', label=r'$\beta=10^{-3}$')
ax_scatter.set_xlabel('RV volume [mL]')
ax_scatter.set_ylabel('LV volume [mL]')
ax_scatter.legend()

### Top KDE -> For RV
sns.kdeplot(x=rv_vol_real, ax=ax_histx, fill=True, color='tab:grey')
sns.kdeplot(x=rv_vol_nf, ax=ax_histx, fill=True, color='aqua')
sns.kdeplot(x=rv_vol_vae2, ax=ax_histx, fill=True, color='tab:orange')
sns.kdeplot(x=rv_vol_vae3, ax=ax_histx2, fill=True, color='tab:red')

### Right KDE -> For LV
sns.kdeplot(y=lv_vol_real, ax=ax_histy, fill=True, color='tab:grey')
sns.kdeplot(y=lv_vol_nf, ax=ax_histy, fill=True, color='aqua')
sns.kdeplot(y=lv_vol_vae2, ax=ax_histy, fill=True, color='tab:orange')
sns.kdeplot(y=lv_vol_vae3, ax=ax_histy2, fill=True, color='tab:red')

# --- Hide tick labels and axis labels for KDE plots ---
ax_histx.tick_params(axis='x', labelbottom=False)
ax_histx.tick_params(axis='y', left=False)
ax_histx.set_xlabel('')  # remove xlabel
ax_histx.set_ylabel('')  # remove ylabel
ax_histx.set_yticks([])

ax_histy.tick_params(axis='y', labelleft=False)
ax_histy.tick_params(axis='x', bottom=False)
ax_histy.set_xlabel('')
ax_histy.set_ylabel('')
ax_histy.set_xticks([])

ax_histx2.set_yticks([])
ax_histx2.set_ylabel('')
ax_histx2.spines['right'].set_visible(False)
ax_histx2.spines['top'].set_visible(False)

ax_histy2.set_xticks([])
ax_histy2.set_xlabel('')
ax_histy2.spines['top'].set_visible(False)
ax_histy2.spines['right'].set_visible(False)

# --- Remove spines for marginal plots ---
for ax in [ax_histx, ax_histy, ax_histx2, ax_histy2]:
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)


# Define the folder path
folder_save = "../figures_experiments"

# Create the folder
os.makedirs(folder_save + "/compare_clinical_metrics", exist_ok=True)
plt.show()
# plt.savefig(folder_save + "/compare_clinical_metrics" + "/rv_lv_scatter.pdf")






### Figure 2: 1d distributions of myocardial mass, LVEDV, RVEDV
myo_mass_real = df_real['Myo_Mass_g']
myo_mass_nf = df_nf['Myo_Mass_g']
myo_mass_vae2 = df_vae2['Myo_Mass_g']
myo_mass_vae3 = df_vae3['Myo_Mass_g']


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

sns.kdeplot(myo_mass_real, ax=ax1, color='tab:grey', fill=True, alpha=0.3, label='real', bw_adjust=1.0)
sns.kdeplot(myo_mass_nf, ax=ax1, color='aqua', fill=True, alpha=0.3, label='NF', bw_adjust=1.0)
sns.kdeplot(myo_mass_vae2, ax=ax1, color='tab:orange', fill=True, alpha=0.3, label=r'$\beta=10^{-2}$', bw_adjust=1.0)
sns.kdeplot(myo_mass_vae3, ax=ax2, color='tab:red', fill=True, alpha=0.3, label=r'$\beta=10^{-3}$', bw_adjust=1.0)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2)
ax1.set_xlabel('myocardial mass [g]')
ax1.set_ylabel('density')
ax2.set_ylabel('density', color='tab:red')
ax2.tick_params(axis='y', colors='tab:red')
ax2.spines['right'].set_visible(False)
# plt.savefig(folder_save + "/compare_clinical_metrics" + "/myocardial_mass.pdf")
plt.show()





### Figure 3: 1d distributions of LVEDV, RVEDV
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

sns.kdeplot(rv_vol_real, ax=ax1, color='tab:grey', fill=True, alpha=0.3, label='real')
sns.kdeplot(rv_vol_nf, ax=ax1, color='aqua', fill=True, alpha=0.3, label='NF')
sns.kdeplot(rv_vol_vae2, ax=ax1, color='tab:orange', fill=True, alpha=0.3, label=r'$\beta=10^{-2}$')
sns.kdeplot(rv_vol_vae3, ax=ax2, color='tab:red', fill=True, alpha=0.3, label=r'$\beta=10^{-3}$')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2)
ax1.set_xlabel('RV volume [mL]')
ax1.set_ylabel('density')
ax2.set_ylabel('density', color='tab:red')
ax2.tick_params(axis='y', colors='tab:red')
ax2.spines['right'].set_visible(False)
# plt.savefig(folder_save + "/compare_clinical_metrics" + "/rvedv.pdf")
plt.show()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

sns.kdeplot(lv_vol_real, ax=ax1, color='tab:grey', fill=True, alpha=0.3, label='real')
sns.kdeplot(lv_vol_nf, ax=ax1, color='aqua', fill=True, alpha=0.3, label='NF')
sns.kdeplot(lv_vol_vae2, ax=ax1, color='tab:orange', fill=True, alpha=0.3, label=r'$\beta=10^{-2}$')
sns.kdeplot(lv_vol_vae3, ax=ax2, color='tab:red', fill=True, alpha=0.3, label=r'$\beta=10^{-3}$')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2)
ax1.set_xlabel('LV volume [mL]')
ax1.set_ylabel('density')
ax2.set_ylabel('density', color='tab:red')
ax2.tick_params(axis='y', colors='tab:red')
ax2.spines['right'].set_visible(False)
# plt.savefig(folder_save + "/compare_clinical_metrics" + "/lvedv.pdf")
# plt.close()
plt.show()






# ### Table result -- print wasserstein distance and KL divergence for Myo_Mass, LVEDV, RVEDV for all models
# ### Figure 4 --- bar chart of WD for different models and different biomarkers
# biomarkers = ['RV_Vol_mL', 'LV_Vol_mL', 'Myo_Mass_g']
#
# def add_labels(bars):
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(
#             bar.get_x() + bar.get_width() / 2,
#             height,
#             f'{height:.2f}',
#             ha='center',
#             va='bottom',
#             fontsize=14
#         )
#
# biomarkers_diffs = []
# for biom in biomarkers:
#     x = df_real[biom]
#
#     y_nf = df_nf[biom]
#     y_vae = df_vae[biom]
#     y_gan = df_gan[biom]
#
#     wass_nf = wasserstein_distance(x, y_nf)
#     wass_vae = wasserstein_distance(x, y_vae)
#     wass_gan = wasserstein_distance(x, y_gan)
#
#
#     kl_nf = kl_div_histogram(x=x, y=y_nf)
#     kl_vae = kl_div_histogram(x=x, y=y_vae)
#     kl_gan = kl_div_histogram(x=x, y=y_gan)
#
#     biomarkers_diffs.append([wass_nf, wass_vae, wass_gan, kl_nf, kl_vae, kl_gan])
#
#     print(f"{biom}  \n"
#           f"NF_Wass: {wass_nf},  NF_KL: {kl_nf}\n"
#           f"VAE_Wass: {wass_vae}, VAE_KL: {kl_vae}\n"
#           f"GAN_Wass: {wass_gan}, GAN_KL: {kl_gan}")
#
#
#     categories = ['CaN_Do', 'VAE', 'GAN']
#
#     wass = [wass_nf, wass_vae, wass_gan]
#     kl = [kl_nf, kl_vae, kl_gan]
#
#     x = np.arange(len(categories))  # positions of main categories
#     width = 0.25  # width of each bar
#
#     plt.figure()
#     fig_wass = plt.bar(x - width, wass, width, color='tab:blue', label='Wasserstein distance')
#     fig_kl = plt.bar(x, kl, width, color='tab:red', label='KL divergence')
#
#     plt.xticks(x, categories)
#     plt.xlabel('')
#     plt.ylabel('')
#     plt.legend()
#
#     add_labels(fig_wass)
#     add_labels(fig_kl)
#
#     plt.tight_layout()
#     plt.savefig(f'../figures_experiments/compare_clinical_metrics/bar_{biom}.pdf')
#
#
#
#
#
#
# ### Figure 5 --- "radar" charts of WD and KL for different models
# categories = ['LVEDV', 'RVEDV', 'myocardial mass']
#
# nf_wd = [biomarkers_diffs[i][0] for i in range(len(categories))]
# vae_wd = [biomarkers_diffs[i][1] for i in range(len(categories))]
# gan_wd = [biomarkers_diffs[i][2] for i in range(len(categories))]
#
# nf_kl = [biomarkers_diffs[i][3] for i in range(len(categories))]
# vae_kl = [biomarkers_diffs[i][4] for i in range(len(categories))]
# gan_kl = [biomarkers_diffs[i][5] for i in range(len(categories))]
#
#
# # Close the loop
# nf_wd += nf_wd[:1]
# vae_wd += vae_wd[:1]
# gan_wd += gan_wd[:1]
# nf_kl += nf_kl[:1]
# vae_kl += vae_kl[:1]
# gan_kl += gan_kl[:1]
#
#
# angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
# angles += angles[:1]  # match data length
#
# fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
#
# # Plot each dataset
# ax.plot(angles, nf_wd, linewidth=2, linestyle='solid', color='aqua', label='NF')
# ax.fill(angles, nf_wd, color='aqua', alpha=0.1)
#
# ax.plot(angles, vae_wd, linewidth=2, linestyle='solid', color='tab:orange', label='VAE')
# ax.fill(angles, vae_wd, color='tab:orange', alpha=0.1)
#
# ax.plot(angles, gan_wd, linewidth=2, linestyle='solid', color='tab:red', label='GAN')
# ax.fill(angles, gan_wd, color='tab:red', alpha=0.1)
# ax.set_xticks(np.linspace(0, 2*np.pi, len(categories), endpoint=False))
# ax.set_xticklabels(categories)
# ax.tick_params(axis='x', pad=15)
# plt.savefig('../figures_experiments/compare_clinical_metrics/wd_radar.pdf')
#
#
#
# fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
#
# # Plot each dataset
# ax.plot(angles, nf_kl, linewidth=2, linestyle='solid', color='aqua', label='NF')
# ax.fill(angles, nf_kl, color='aqua', alpha=0.1)
#
# ax.plot(angles, vae_kl, linewidth=2, linestyle='solid', color='tab:orange', label='VAE')
# ax.fill(angles, vae_kl, color='tab:orange', alpha=0.1)
#
# ax.plot(angles, gan_kl, linewidth=2, linestyle='solid', color='tab:red', label='GAN')
# ax.fill(angles, gan_kl, color='tab:red', alpha=0.1)
#
# ax.set_xticks(np.linspace(0, 2*np.pi, len(categories), endpoint=False))
# ax.set_xticklabels(categories)
# ax.tick_params(axis='x', pad=15)
# plt.savefig('../figures_experiments/compare_clinical_metrics/kl_radar.pdf')
