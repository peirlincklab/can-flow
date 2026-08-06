import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams, font_manager
import os

params = {
            'axes.labelsize': 35,
            'font.size': 35,
            'legend.fontsize': 35,
            'xtick.labelsize': 35,
            'ytick.labelsize': 35,
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

os.makedirs('../figures_experiments/ablation', exist_ok=True)

dims = ['10', '20', '30', '40', '50']
dims_vals = [10, 20, 30, 40, 50]
biomarkers = ['LV_Vol_mL', 'RV_Vol_mL', 'Myo_Mass_g', 'RVEDV_LVEDV_ratio', 'Long_axis_length', 'LV_Sphericity']


### find the min & max of KL div for shared y-axis
kl_minmax_canflow = []
kl_minmax_cvae2 = []
kl_minmax_cvae3 = []

for biom in biomarkers:
    for dim in dims:
        with open(f"../ablation/kl_dict_latentdim_{dim}.pkl", "rb") as f:
            pkl_data = pickle.load(f)

        data = pkl_data[biom]
        kl_minmax_canflow.append(data[0])
        kl_minmax_cvae2.append(data[1])
        kl_minmax_cvae3.append(data[2])

min_canflow = min(kl_minmax_canflow)
max_canflow = max(kl_minmax_canflow)

min_cvae2 = min(kl_minmax_cvae2)
max_cvae2 = max(kl_minmax_cvae2)

min_cvae3 = min(kl_minmax_cvae3)
max_cvae3 = max(kl_minmax_cvae3)

min_all = min([min_canflow, min_cvae2, min_cvae3])
max_all = max([max_canflow, max_cvae2, max_cvae3])



for biom in biomarkers:

    kl_canflow = []
    kl_cvae2 = []
    kl_cvae3 = []
    for dim in dims:
        with open(f"../ablation/kl_dict_latentdim_{dim}.pkl", "rb") as f:
            pkl_data = pickle.load(f)

        data = pkl_data[biom]
        kl_canflow.append(data[0])
        kl_cvae2.append(data[1])
        kl_cvae3.append(data[2])


    plt.figure(figsize=(9, 7))
    plt.plot(dims_vals, kl_canflow, '-o', linewidth=4, markersize=10,color='#D55E00')
    plt.plot(dims_vals, kl_cvae2, '-o', linewidth=4, markersize=10,color='#0072B2')
    plt.plot(dims_vals, kl_cvae3, '-o', linewidth=4, markersize=10,color='#56B4E9')

    if biom not in ['LV_Vol_mL', 'RV_Vol_mL', 'Myo_Mass_g']:
        plt.xlabel('latent dimensionality')
        plt.xticks(dims_vals)
        plt.tick_params(axis='x', length=12, width=4)
    else:
        plt.xticks(dims_vals)
        plt.tick_params(axis='x', length=12, width=4, labelbottom=False)

    if biom not in ['LV_Vol_mL', 'RVEDV_LVEDV_ratio']:
        plt.tick_params(axis='y', length=12, width=4, labelleft=False)
    else:
        plt.tick_params(axis='y', length=12, width=4)

    plt.ylim(min_all * 0.85, max_all * 1.15)
    plt.yscale('log')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'../figures_experiments/ablation/{biom}_scatter.svg', bbox_inches="tight")
