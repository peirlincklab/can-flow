import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
import seaborn as sns

### This file is used to generate the results illustrated in Figure 5 of the manuscript
### We can choose to visualize different clinical phenotypes apart from LVEDV and RVEDV


def generate_plot(df, df_str, color_plot, xmin, xmax, ymin, ymax, df_real_overlay=None):
    """
        Generate a scatter plot of a clinical phenotype versus another phenotype with marginal KDE distributions.

        This function creates a joint-style plot showing the relationship between
        two clinical phenotypes. The main panel contains the scatter plot, while the top and
        right panels show the marginal kernel density estimates (KDEs) for the two clinical phenotypes

        Optionally, the marginal KDE distributions of a real dataset can be overlaid
        in black for comparison with a generated dataset.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing all clinical phenotypes

        df_str : str
            Name of the dataset or model being plotted. This string is used when
            saving the figure.

            If `df_str == 'real'`, the y-axis label is shown. Otherwise, only the
            x-axis label is shown.

        color_plot : str
            Color used for the scatter points and KDE distributions of `df`.

        xmin : float
            Minimum value of the x-axis.

        xmax : float
            Maximum value of the x-axis.

        ymin : float
            Minimum value of the y-axis.

        ymax : float
            Maximum value of the y-axis.

        df_real_overlay : pandas.DataFrame, optional
            Optional real-data DataFrame used to overlay marginal KDE distributions
            on top of the KDEs of `df`.
        """
    fig = plt.figure(figsize=(10.5, 8.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          wspace=0.02, hspace=0.02)

    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

    ax_histx2 = ax_histx.twinx()
    ax_histy2 = ax_histy.twiny()

    ax_scatter.scatter(df['Myo_Mass_g'], df['RV_Vol_mL'], s=150, color=color_plot, alpha=0.8, edgecolors='white', linewidth=0.2)

    if df_str == 'real':
        ax_scatter.set_ylabel('RV volume [mL]')
    ax_scatter.set_xlabel('myocardial mass [g]')

    ax_scatter.set_xlim([xmin, xmax])
    ax_scatter.set_ylim([ymin, ymax])

    ax_scatter.set_yticks([100, 150, 200, 250])
    ax_scatter.set_xticks([80, 120, 160, 200, 240])

    ### Top KDE -> For RV
    sns.kdeplot(x=df['Myo_Mass_g'], ax=ax_histx, fill=True, color=color_plot)

    ### Right KDE -> For LV
    sns.kdeplot(y=df['RV_Vol_mL'], ax=ax_histy, fill=True, color=color_plot)

    if df_real_overlay is not None:
        sns.kdeplot(x=df_real_overlay['Myo_Mass_g'], ax=ax_histx, fill=True, color='black', alpha=0.2)
        sns.kdeplot(y=df_real_overlay['RV_Vol_mL'], ax=ax_histy, fill=True, color='black', alpha=0.2)

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

    for ax in [ax_histx, ax_histy, ax_histx2, ax_histy2]:
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(False)

    plt.savefig(f'../figures_experiments/compare_clinical_metrics/{df_str}_scatter.svg', bbox_inches="tight")

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

frac_train = 0.7
NumAll = 2208

NumTrainSamples = int(NumAll * frac_train)

### we visualize results only for CAN-FLOW and the two most performant cVAEs
df_real = pd.read_pickle('../data_models_saved/data/dataframes_clinical_info/df_real_clinical.pkl')

df_nf = pd.read_pickle('../data_models_saved/data/dataframes_clinical_info/df_nf_clinical.pkl')
df_vae2 = pd.read_pickle('../data_models_saved/data/dataframes_clinical_info/df_vae2_clinical.pkl')
df_vae3 = pd.read_pickle('../data_models_saved/data/dataframes_clinical_info/df_vae3_clinical.pkl')

xmin = min([df_real['Myo_Mass_g'].min(), df_nf['Myo_Mass_g'].min(), df_vae2['Myo_Mass_g'].min(), df_vae3['Myo_Mass_g'].min()]) - 10
xmax = max([df_real['Myo_Mass_g'].max(), df_nf['Myo_Mass_g'].max(), df_vae2['Myo_Mass_g'].max(), df_vae3['Myo_Mass_g'].max()]) + 10

ymin = min([df_real['RV_Vol_mL'].min(), df_nf['RV_Vol_mL'].min(), df_vae2['RV_Vol_mL'].min(), df_vae3['RV_Vol_mL'].min()]) - 10
ymax = max([df_real['RV_Vol_mL'].max(), df_nf['RV_Vol_mL'].max(), df_vae2['RV_Vol_mL'].max(), df_vae3['RV_Vol_mL'].max()]) + 10

models = ['real', 'can_flow', 'cvae2', 'cvae3']
dfs = [df_real, df_nf, df_vae2, df_vae3]
colors = ['black', '#D55E00', '#0072B2', '#56B4E9']

for i, mod in enumerate(models):

    if i ==0:
        df_real_overlay = None
    else:
        df_real_overlay = df_real

    generate_plot(df=dfs[i], df_str=mod, color_plot=colors[i],
                  xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, df_real_overlay=df_real_overlay)





