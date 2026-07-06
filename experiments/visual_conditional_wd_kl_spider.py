import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
import os


### This file generates part of the results illustrated in Figure 6 of the manuscript


def plot_spider_metadata(metadata_group1, metadata_group2, name_plot, colors_circ, fontsize_yaxis=15):
    """
        Generate a two-bin spider plot for comparing metadata-specific results.

        This function creates a polar plot divided into two semi-circles. The top
        semi-circle represents the first metadata group, while the bottom semi-circle
        represents the second metadata group. For each group, the function compares
        three generative models across six anatomical/phenotype metrics.

        The plotted models are:

        - CAN-FLWO
        - cVAE with beta = 10^{-2}
        - cVAE with beta = 10^{-3}

        The metrics shown in the spider plot are:

        - RVEDV
        - LVEDV
        - myocardial mass
        - RVEDV / LVEDV
        - long-axis length
        - sphericity

        Parameters
        ----------
        metadata_group1 : numpy.ndarray
            Array containing the metric values for the first metadata group.

            The function uses the following columns:

            - column 0: CAN-DO
            - column 2: cVAE beta = 10^{-2}
            - column 3: cVAE beta = 10^{-3}

        metadata_group2 : numpy.ndarray
            Array containing the metric values for the second metadata group.


        name_plot : str
            Name used when saving the output figure

        colors_circ : list or tuple of str
            Colors used for the outer semi-circle boundaries.

            - `colors_circ[0]` is used for the top semi-circle.
            - `colors_circ[1]` is used for the bottom semi-circle.

        fontsize_yaxis : int, optional
            Font size of the radial axis tick labels. Default is 15.
        """


    categories = ['RVEDV', 'LVEDV', "myocardial\nmass", r"$\frac{RVEDV}{LVEDV}$", "long-axis\nlength", "sphericity"]

    models = ["CAN-FLOW", r"cVAE $\beta=10^{-2}$", r"cVAE $\beta=10^{-3}$"]

    colors = ['#D55E00', '#0072B2', '#56B4E9']

    data_top_group1 = {
        "CAN-FLOW": metadata_group1[:, 0],
        r"cVAE $\beta=10^{-2}$": metadata_group1[:, 2],
        r"cVAE $\beta=10^{-3}$": metadata_group1[:, 3],
    }

    data_bottom_group2 = {
        "CAN-FLOW": metadata_group2[:, 0],
        r"cVAE $\beta=10^{-2}$": metadata_group2[:, 2],
        r"cVAE $\beta=10^{-3}$": metadata_group2[:, 3],
    }

    all_values = []

    for model in models:
        all_values.extend(data_top_group1[model])
        all_values.extend(data_bottom_group2[model])

    max_value = np.max(all_values)

    # Add some margin
    circle_radius = 1.15 * max_value


    ### angles
    angles_top_deg = np.linspace(10, 170, len(categories))
    angles_top = np.deg2rad(angles_top_deg)

    angles_bottom_deg = np.linspace(190, 350, len(categories))
    angles_bottom = np.deg2rad(angles_bottom_deg)

    ### figure
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)


    ax.spines["polar"].set_visible(False)

    ax.set_xticks([])

    ### radial ticks
    ax.set_ylim(0, circle_radius * 1.15)

    ticks = np.linspace(0, circle_radius, 5)[1:]

    ax.set_yticks(ticks)
    ax.set_yticklabels(
        [f"{t:.2f}" for t in ticks],
        fontsize=fontsize_yaxis
    )

    ax.set_rlabel_position(0)

    theta_top_boundary = np.linspace(0, np.pi, 500)
    theta_bottom_boundary = np.linspace(np.pi, 2 * np.pi, 500)

    ax.plot(
        theta_top_boundary,
        circle_radius * np.ones_like(theta_top_boundary),
        color=colors_circ[0],
        linewidth=4,
        solid_capstyle="round",
    )

    ax.plot(
        theta_bottom_boundary,
        circle_radius * np.ones_like(theta_top_boundary),
        color=colors_circ[1],
        linewidth=4,
        solid_capstyle="round"
    )


    ax.plot(
        [0, np.pi],
        [circle_radius, circle_radius],
        color="black",
        linewidth=1.5
    )


    for angle in angles_top:
        ax.plot([angle, angle], [0, circle_radius], color="gray", linewidth=0.8, alpha=0.5)

    for angle in angles_bottom:
        ax.plot([angle, angle], [0, circle_radius], color="gray", linewidth=0.8, alpha=0.5)



    for model, color in zip(models, colors):
        values = np.array(data_top_group1[model])

        ### close semi-triangle through the center
        theta = np.concatenate(([angles_top[0]], angles_top, [angles_top[-1]]))
        r = np.concatenate(([0], values, [0]))

        ax.plot(theta, r, linewidth=2, label=f"{model} - top", color=color)
        ax.fill(theta, r, alpha=0.12, color=color)


    for model, color in zip(models, colors):
        values = np.array(data_bottom_group2[model])

        theta = np.concatenate(([angles_bottom[0]], angles_bottom, [angles_bottom[-1]]))
        r = np.concatenate(([0], values, [0]))

        ax.plot(theta, r, linewidth=2, label=f"{model} - bottom", color=color)
        ax.fill(theta, r, alpha=0.12, color=color)


    ### category labels
    for angle, label in zip(angles_top, categories):
        ax.text(
            angle,
            circle_radius * 1.25,
            label,
            ha="center",
            va="center"
        )

    for angle, label in zip(angles_bottom, categories):
        ax.text(
            angle,
            circle_radius * 1.25,
            label,
            ha="center",
            va="center"
        )


    ax.text(
        np.deg2rad(90),
        1.22,
        "",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold"
    )

    ax.text(
        np.deg2rad(270),
        1.22,
        "",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(f'../figures_experiments/conditional_wd_kl_heatmaps/{name_plot}_spider_2bins.svg')




os.makedirs('../figures_experiments/conditional_wd_kl_heatmaps', exist_ok=True)

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


wass_kl_metrics = np.load('../data_models_saved/data/conditional_wd_kl_forspider_2bins.npy', allow_pickle=True)

models = ['CAN-FLOW', r'$\beta=10^{-1}$', r'$\beta=10^{-2}$', r'$\beta=10^{-3}$', r'$\beta=10^{-4}$', r'$\beta=10^{-5}$', r'$\beta=10^{-6}$']
clinical_phenotypes = ['RVEDV', 'LVEDV', 'myocardial\nmass', "RV_LV_ratio", "LA_length", "Spher"]

n_models = len(models)
n_phenos = len(clinical_phenotypes)

### arrange the data according to the files
wass_female = np.array([
    [wass_kl_metrics[j][0][i][0][0] for j in range(n_models)]
    for i in range(n_phenos)
], dtype=float)

wass_male = np.array([
    [wass_kl_metrics[j][0][i][0][1] for j in range(n_models)]
    for i in range(n_phenos)
], dtype=float)


kl_female = np.array([
    [wass_kl_metrics[j][1][i][0][0] for j in range(n_models)]
    for i in range(n_phenos)
], dtype=float)

kl_male = np.array([
    [wass_kl_metrics[j][1][i][0][1] for j in range(n_models)]
    for i in range(n_phenos)
], dtype=float)



plot_spider_metadata(metadata_group1=wass_male, metadata_group2=wass_female, colors_circ=["teal", "purple"], name_plot='sex_wass')
plot_spider_metadata(metadata_group1=kl_male, metadata_group2=kl_female, colors_circ=["teal", "purple"], name_plot='sex_kl')

wass_bin1 = np.array([
    [wass_kl_metrics[j][0][i][1][0] for j in range(n_models)]
    for i in range(n_phenos)
], dtype=float)

wass_bin2 = np.array([
    [wass_kl_metrics[j][0][i][1][1] for j in range(n_models)]
    for i in range(n_phenos)
], dtype=float)


kl_bin1 = np.array([
    [wass_kl_metrics[j][1][i][1][0] for j in range(n_models)]
    for i in range(n_phenos)
], dtype=float)

kl_bin2 = np.array([
    [wass_kl_metrics[j][1][i][1][1] for j in range(n_models)]
    for i in range(n_phenos)
], dtype=float)


plot_spider_metadata(metadata_group1=wass_bin1, metadata_group2=wass_bin2, colors_circ=["sandybrown", "saddlebrown"], name_plot='age_wass')
plot_spider_metadata(metadata_group1=kl_bin1, metadata_group2=kl_bin2, colors_circ=["sandybrown", "saddlebrown"], name_plot='age_kl')






