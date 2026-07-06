### Visualize Cov and MMD
### We can choose to visualize all results from cVAEs, or from the two most performant ones
### Currently we visualize results for CAN-FLOW and all trained cVAEs

### This file is used to generate the results illustrated in Figure 3 and B.11


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
import os

os.makedirs('../figures_experiments/visual_mmd_cov_wd_kl', exist_ok=True)


params = {
            'axes.labelsize': 28,
            'font.size': 28,
            'legend.fontsize': 28,
            'xtick.labelsize': 28,
            'ytick.labelsize': 28,
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


mmd = np.load("../data_models_saved/data/mmd_vals.npy")
cov = np.load("../data_models_saved/data/cov_vals.npy")

metrics_str = ['Cov (%)', 'MMD']
metrics = [cov, mmd]

for metric, metric_str in zip(metrics, metrics_str):

    x = np.array([0, 2, 4, 6, 8, 10, 12])

    if metric_str == 'Cov (%)':

        means = {
              'CAN-FLOW': 100 * np.mean(metric[:, 0], axis=0),
              '   cVAE': 100 * np.mean(metric[:, 1], axis=0),
              '  cVAE  ': 100 * np.mean(metric[:, 2], axis=0),
              'cVAE ': 100 * np.mean(metric[:, 3], axis=0),
              'cVAE     ': 100 * np.mean(metric[:, 4], axis=0),
              'cVAE      ': 100 * np.mean(metric[:, 5], axis=0),
              'cVAE': 100 * np.mean(metric[:, 6], axis=0)
        }

        stds = {
              'CAN-FLOW': 100 * np.std(metric[:, 0], axis=0),
              '   cVAE': 100 * np.std(metric[:, 1], axis=0),
              '  cVAE  ': 100 * np.std(metric[:, 2], axis=0),
              'cVAE ': 100 * np.std(metric[:, 3], axis=0),
              'cVAE     ': 100 * np.std(metric[:, 4], axis=0),
              'cVAE      ': 100 * np.std(metric[:, 5], axis=0),
              'cVAE': 100 * np.std(metric[:, 6], axis=0)
        }
    else:
        means = {
            'CAN-FLOW': np.mean(metric[:, 0], axis=0),
            '   cVAE': np.mean(metric[:, 1], axis=0),
            '  cVAE  ': np.mean(metric[:, 2], axis=0),
            'cVAE ': np.mean(metric[:, 3], axis=0),
            'cVAE     ': np.mean(metric[:, 4], axis=0),
            'cVAE      ': np.mean(metric[:, 5], axis=0),
            'cVAE': np.mean(metric[:, 6], axis=0)
        }

        stds = {
            'CAN-FLOW': np.std(metric[:, 0], axis=0),
            '   cVAE': np.std(metric[:, 1], axis=0),
              '   cVAE  ': np.std(metric[:, 2], axis=0),
              'cVAE ': np.std(metric[:, 3], axis=0),
              'cVAE     ':  np.std(metric[:, 4], axis=0),
              'cVAE      ': np.std(metric[:, 5], axis=0),
              'cVAE': np.std(metric[:, 6], axis=0)
        }


    labels = list(means.keys())
    mean_values = list(means.values())
    std_values = list(stds.values())

    colors_models = ['#D55E00', "#004C7A", '#0072B2', '#56B4E9', "#7EC7F0", "#A6DDF5", "#D0EFFB"]


    plt.figure(figsize=(12, 5))

    bars = plt.bar(
        x,
        mean_values,
        yerr=std_values,
        capsize=12,
        color=colors_models,
    )

    plt.xticks(x, labels)
    plt.ylabel(metric_str)

    plt.grid(alpha=0.5)

    if metric_str =='Cov (%)':
      plt.yticks([10, 20, 30, 40, 50])
    else:
      plt.yticks([2, 4, 6, 8, 10])

    plt.tight_layout()
    # plt.yscale('log')
    plt.savefig(f'../figures_experiments/visual_mmd_cov_wd_kl/{metric_str}_overall_all_cvaes.svg')
    # plt.show()