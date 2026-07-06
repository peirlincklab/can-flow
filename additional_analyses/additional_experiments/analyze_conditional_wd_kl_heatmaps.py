import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
import os


### Plots the heatmaps of KL and WD

os.makedirs('../../figures_experiments/conditional_wd_kl_heatmaps', exist_ok=True)

params = {
            'axes.labelsize': 40,
            'font.size': 40,
            'legend.fontsize': 40,
            'xtick.labelsize': 40,
            'ytick.labelsize': 40,
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


wass_kl_metrics = np.load('../../data_models_saved/data/conditional_wd_kl.npy')

models = ['', r'', r'']
clinical_phenotypes = ['', '', '']


# ============================================================
# Build arrays
# Shapes: (n_phenos, n_models)
# ==========================================================
ind_models = [0, 2, 3]
n_phenos = len(clinical_phenotypes)

wass_sex = np.array([
    [wass_kl_metrics[j][0][i][0] for j in ind_models]
    for i in range(n_phenos)
], dtype=float)

kl_sex = np.array([
    [wass_kl_metrics[j][1][i][0] for j in ind_models]
    for i in range(n_phenos)
], dtype=float)

wass_age = np.array([
    [wass_kl_metrics[j][0][i][1] for j in ind_models]
    for i in range(n_phenos)
], dtype=float)

kl_age = np.array([
    [wass_kl_metrics[j][1][i][1] for j in ind_models]
    for i in range(n_phenos)
], dtype=float)

# Heatmaps: rows = models, cols = phenotypes
wass_age_hm = wass_age.T
wass_sex_hm = wass_sex.T
kl_age_hm = kl_age.T
kl_sex_hm = kl_sex.T

# Consistent scales within each metric family
wd_vmin = min(wass_age_hm.min(), wass_sex_hm.min())
wd_vmax = max(wass_age_hm.max(), wass_sex_hm.max())

kl_vmin = min(kl_age_hm.min(), kl_sex_hm.min())
kl_vmax = max(kl_age_hm.max(), kl_sex_hm.max())


# ============================================================
# Helper functions
# ============================================================
def get_best_rows_per_column(data):
    """
    data: shape (n_models, n_phenos)
    Returns a list with the best row index/indices for each column.
    Handles ties.
    """
    best_rows_per_col = []

    for j in range(data.shape[1]):
        col = data[:, j]
        min_val = np.min(col)
        best_rows = np.where(np.isclose(col, min_val))[0].tolist()
        best_rows_per_col.append(best_rows)

    return best_rows_per_col


def plot_heatmap_with_bold_best(
    ax,
    data,
    row_labels,
    col_labels,
    cmap="viridis",
    vmin=None,
    vmax=None,
    cbar_label=None,
    fmt="{:.2e}"
):
    """
    data: shape (n_models, n_phenos)
    Lower values are better.
    Best value(s) in each column are bolded.
    Dark cells get white text, light cells get black text.
    """
    im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)

    best_rows_per_col = get_best_rows_per_column(data)
    norm = im.norm

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)


    # Annotate cells
    for i in range(data.shape[0]):        # rows = models
        for j in range(data.shape[1]):    # cols = phenotypes
            val = data[i, j]
            is_best = i in best_rows_per_col[j]

            text_color = 'white' if norm(val) > 0.5 else 'black'

            ax.text(
                j, i, fmt.format(val),
                ha='center', va='center',
                fontsize=35,
                color=text_color,
                fontweight='bold' if is_best else 'normal'
            )

    return im


# ============================================================
# Plot 4 separate figures
# ============================================================

fig, ax = plt.subplots(figsize=(7, 6))
plot_heatmap_with_bold_best(
    ax,
    wass_age_hm,
    models,
    clinical_phenotypes,
    cmap="YlOrRd",
    vmin=wd_vmin,
    vmax=wd_vmax,
    fmt="{:.4}"
)
plt.tight_layout()
plt.savefig('../figures_experiments/conditional_wd_kl_heatmaps/wass_wrt_age_less_models.svg')


fig, ax = plt.subplots(figsize=(7, 6))
plot_heatmap_with_bold_best(
    ax,
    wass_sex_hm,
    models,
    clinical_phenotypes,
    cmap="YlOrRd",
    vmin=wd_vmin,
    vmax=wd_vmax,
    fmt="{:.4}"
)
plt.tight_layout()
plt.savefig('../figures_experiments/conditional_wd_kl_heatmaps/wass_wrt_sex_less_models.svg')


fig, ax = plt.subplots(figsize=(7, 6))
plot_heatmap_with_bold_best(
    ax,
    kl_age_hm,
    models,
    clinical_phenotypes,
    cmap="Blues",
    vmin=kl_vmin,
    vmax=kl_vmax,
    fmt="{:.4}"
)
plt.tight_layout()
plt.savefig('../figures_experiments/conditional_wd_kl_heatmaps/kl_wrt_age_less_models.svg')


fig, ax = plt.subplots(figsize=(7, 6))
plot_heatmap_with_bold_best(
    ax,
    kl_sex_hm,
    models,
    clinical_phenotypes,
    cmap="Blues",
    vmin=kl_vmin,
    vmax=kl_vmax,
    fmt="{:.4}"
)
plt.tight_layout()
plt.savefig('../figures_experiments/conditional_wd_kl_heatmaps/kl_wrt_sex_less_models.svg')