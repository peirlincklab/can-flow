import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager

### This file generates the results illustrated in Figure 5, 6,, and B.12

### We can choose to visualize information for all cVAEs, or only for the most performant ones.
### Currently, we this file visualzises results from all cVAEs

def spider_plot(models, colors, name_plot):
    """
    Generate a spider plot comparing multiple models across WD/KL div. of clinical phenotypes.

    This function creates a polar spider plot where each axis corresponds to one
    anatomical or clinical phenotype (WD or KL divergence). Each model is represented by a closed polygon.
    Currently, radial axis is shown in logarithmic scale. This can change to linear axis

    The metrics shown in the plot are:

    - RVEDV
    - LVEDV
    - myocardial mass
    - RVEDV / LVEDV
    - long-axis length
    - sphericity

    Parameters
    ----------
    models : list of numpy.ndarray
        List containing the metric values for each model.

    colors : list of str
        List of colors used for the model curves and filled regions.

    name_plot : str
        Name used when saving the output figure. The figure is saved as:
    """
    
    categories = ["RVEDV", "LVEDV", "myocardial\nmass", r"$\frac{RVEDV}{LVEDV}$", "long-axis\nlength", "sphericity"]
    N = len(categories)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    for i in range(len(models)):
        models[i] = np.concatenate([models[i], [models[i][0]]])


    max_value = np.max([np.max(models[0]),  np.max(models[1]), np.max(models[2]),
                        np.max(models[3]), np.max(models[4]), np.max(models[5]), np.max(models[6])])

    ### add some margin
    circle_radius = 1.1 * max_value

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.spines["polar"].set_linewidth(4)

    ### plot each model
    for i in range(len(models)):
        ax.plot(angles, models[i], linewidth=2, color=colors[i],solid_capstyle="round")
        ax.fill(angles, models[i], alpha=0.15, color=colors[i])

    ax.set_yscale("log")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=20, ha='center', va='center')

    ax.set_ylim(0, circle_radius * 2)

    ticks = np.linspace(0, circle_radius, 10)[1:]
    ticks_only = [ticks[0], ticks[1], ticks[3], ticks[-1]]

    ax.set_yticks(ticks_only)
    ax.set_yticklabels(
        [f"{t:.2f}" for t in ticks_only],
        fontsize=15
    )


    ax.set_rlabel_position(0)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.tick_params(axis='x', pad=40)


    # ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))

    plt.tight_layout()
    plt.savefig(f'../figures_experiments/visual_mmd_cov_wd_kl/{name_plot}_spider_allmodels.svg')





params = {
            'axes.labelsize': 20,
            'font.size': 20,
            'legend.fontsize': 20,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
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

biomarkers_wass = np.load("../data_models_saved/data/biomarkers_wass.npy")
biomarkers_kl = np.load("../data_models_saved/data/biomarkers_kl.npy")


canflow_kl = biomarkers_kl[:, 0]
canflow_wass = biomarkers_wass[:, 0]

cvae1_kl = biomarkers_kl[:, 1]
cvae1_wass = biomarkers_wass[:, 1]

cvae2_kl = biomarkers_kl[:, 2]
cvae2_wass = biomarkers_wass[:, 2]

cvae3_kl = biomarkers_kl[:, 3]
cvae3_wass = biomarkers_wass[:, 3]

cvae4_kl = biomarkers_kl[:, 4]
cvae4_wass = biomarkers_wass[:, 4]

cvae5_kl = biomarkers_kl[:, 5]
cvae5_wass = biomarkers_wass[:, 5]

cvae6_kl = biomarkers_kl[:, 6]
cvae6_wass = biomarkers_wass[:, 6]


models_kl = [canflow_kl, cvae1_kl, cvae2_kl, cvae3_kl, cvae4_kl, cvae5_kl, cvae6_kl]
models_wass = [canflow_wass, cvae1_wass, cvae2_wass, cvae3_wass, cvae4_wass, cvae5_wass, cvae6_wass]

colors = ['#D55E00', "#004C7A", '#0072B2', '#56B4E9', "#7EC7F0", "#A6DDF5", "#D0EFFB"]

spider_plot(models=models_kl, colors=colors, name_plot='kl')
spider_plot(models=models_wass, colors=colors, name_plot='wass')


