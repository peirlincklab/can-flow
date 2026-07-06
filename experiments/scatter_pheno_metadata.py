import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
import seaborn as sns
import os

### This file generates the results illustrated in Figure 7 of the manuscript

params = {
            'axes.labelsize': 40,
            'font.size': 40,
            'legend.fontsize': 40,
            'xtick.labelsize': 40,
            'ytick.labelsize': 40,
            'text.usetex': False,
            'axes.linewidth': 2.5,
            'xtick.major.width': 2.5,
            'ytick.major.width': 2.5,
            'xtick.major.size': 2.5,
            'ytick.major.size': 2.5
        }


plt.rcParams.update(params)
font_path = r'C:\Users\kkevopoulos\AppData\Local\Microsoft\Windows\Fonts\SourceSansPro-Regular.otf'
font_prop = font_manager.FontProperties(fname=font_path)
rcParams['font.family'] = font_prop.get_name()


def generate_plot(df, metadata_case, model_str, metadata_str, xmin, xmax, ymin, ymax, case1_color, case2_color):
    """
        Generate a scatter plot of RV versus LV volumes with marginal KDE distributions.

        This function creates a joint visualization of right-ventricular volume
        (`RV_Vol_mL`) and left-ventricular volume (`LV_Vol_mL`) for two metadata-defined
        subgroups. The main panel shows the scatter plot, while the top and right panels
        show the marginal kernel density estimates (KDEs) for RV and LV volumes,
        respectively.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the phenotype data

        metadata_case : list
            contains the two metadara groups to compare

            - `metadata_case[1]` is plotted using `case1_color`.
            - `metadata_case[0]` is plotted using `case2_color`.

        model_str : str
            Name of the model being plotted. If `model_str == 'real'`, y-axis tick
            labels are shown. For other models, y-axis tick labels are hidden, which is
            useful when arranging multiple plots in a grid.

        metadata_str : str
            Name of the metadata variable used to define the cases

        xmin : float
            Minimum value of the x-axis.

        xmax : float
            Maximum value of the x-axis.

        ymin : float
            Minimum value of the y-axis.

        ymax : float
            Maximum value of the y-axis.

        case1_color : str
            Color used for the subgroup indexed by `metadata_case[1]`.

        case2_color : str
            Color used for the subgroup indexed by `metadata_case[0]`.

        Notes
        -----
        This function assumes that `matplotlib.pyplot`, `seaborn`, and `os` have already
        been imported as:

        ```python
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        ```
        """
    fig = plt.figure(figsize=(9.7, 8.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          wspace=0.01, hspace=0.01)

    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

    ax_histx2 = ax_histx.twinx()
    ax_histy2 = ax_histy.twiny()

    ax_scatter.scatter(df['RV_Vol_mL'][metadata_case[1]], df['LV_Vol_mL'][metadata_case[1]],
                       s=80, color=case1_color, alpha=0.65)
    ax_scatter.scatter(df['RV_Vol_mL'][metadata_case[0]], df['LV_Vol_mL'][metadata_case[0]],
                       s=80, color=case2_color, alpha=0.65)

    if model_str == 'real':
        ax_scatter.tick_params(axis='y', length=12, width=4)

        if metadata_str == 'bmi':
            ax_scatter.tick_params(axis='x', length=12, width=4)
        else:
            ax_scatter.tick_params(axis='x', labelbottom=False, length=12, width=4)

    else:
        ax_scatter.tick_params(axis='y', labelleft=False,length=12, width=4)

        if metadata_str == 'bmi':
            ax_scatter.tick_params(axis='x', length=12, width=4)
        else:
            ax_scatter.tick_params(axis='x', labelbottom=False, length=12, width=4)

    ax_scatter.set_xlim([xmin, xmax])
    ax_scatter.set_ylim([ymin, ymax])

    ### Top KDE -> For RV
    sns.kdeplot(x=df['RV_Vol_mL'][metadata_case[1]], ax=ax_histx, fill=True, color=case1_color)
    sns.kdeplot(x=df['RV_Vol_mL'][metadata_case[0]], ax=ax_histx, fill=True, color=case2_color)

    ### Right KDE -> For LV
    sns.kdeplot(y=df['LV_Vol_mL'][metadata_case[1]], ax=ax_histy, fill=True, color=case1_color)
    sns.kdeplot(y=df['LV_Vol_mL'][metadata_case[0]], ax=ax_histy, fill=True, color=case2_color)

    ax_histx.tick_params(axis='x', labelbottom=False, length=12, width=4)
    ax_histx.tick_params(axis='y', left=False, length=12, width=4)
    ax_histx.set_xlabel('')  # remove xlabel
    ax_histx.set_ylabel('')  # remove ylabel
    ax_histx.set_yticks([])

    ax_histy.tick_params(axis='y', labelleft=False, length=12, width=4)
    ax_histy.tick_params(axis='x', bottom=False, length=12, width=4)
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

    folder_save = "../figures_experiments"

    os.makedirs(folder_save + "/scatter_pheno_conf", exist_ok=True)
    plt.savefig(folder_save + "/scatter_pheno_conf" + f"/{model_str}_{metadata_str}_scatter.svg")



x_confounders = pd.read_excel(r"/home/kevopou1/metadata_final.xlsx")
x_confounders.drop(['Participant ID', 'Height', 'Weight', 'Diastolic BP',
                    'Systolic BP', 'Unnamed: 8', 'Unnamed: 9', 'subject_id'], axis=1, inplace=True)
x_confounders = pd.get_dummies(x_confounders, columns=['Sex'])
x_confounders['Sex_Female'] = x_confounders['Sex_Female'].replace({True: 1, False: 0})
x_confounders['Sex_Male'] = x_confounders['Sex_Male'].replace({True: 1, False: 0})
x_confounders = x_confounders[['BMI', 'Age', 'Sex_Female', 'Sex_Male']]
x_confounders = x_confounders.to_numpy()

x_confounders = np.delete(x_confounders, [1746, 1831], axis=0)

outliers = np.load('../utils/outliers_indices.npy')
mask = np.ones(x_confounders.shape[0], dtype=bool)
mask[outliers] = False

x_confounders = x_confounders[mask]

### Compute indices for real data
### indices according to sex
female_ind_real = np.where(x_confounders[:, 2] == 1)[0]
male_ind_real = np.where(x_confounders[:, 2] == 0)[0]

### indices according to bmi
small_bmi_real = np.where(x_confounders[:, 0] <= 22)[0]
large_bmi_real = np.where(x_confounders[:, 0] > 22)[0]

### indices according to age
small_age_real = np.where(x_confounders[:, 1] <= 61)[0]
large_age_real = np.where(x_confounders[:, 1] > 61)[0]

diff_plots_real = [[female_ind_real, male_ind_real], [small_bmi_real, large_bmi_real], [small_age_real, large_age_real]]
diff_plots_str = ['sex', 'bmi', 'age']

df_real = pd.read_pickle('../data_models_saved/data/dataframes_clinical_info/df_real_clinical.pkl')
df_nf = pd.read_pickle(f'../data_models_saved/data/dataframes_clinical_info/df_nf_clinical.pkl')
df_vae2 = pd.read_pickle(f'../data_models_saved/data/dataframes_clinical_info/df_vae2_clinical.pkl')
df_vae3 = pd.read_pickle(f'../data_models_saved/data/dataframes_clinical_info/df_vae3_clinical.pkl')

xmin = min([df_real['RV_Vol_mL'].min(), df_nf['RV_Vol_mL'].min(), df_vae2['RV_Vol_mL'].min(), df_vae3['RV_Vol_mL'].min()]) - 10
xmax = max([df_real['RV_Vol_mL'].max(), df_nf['RV_Vol_mL'].max(), df_vae2['RV_Vol_mL'].max(), df_vae3['RV_Vol_mL'].max()]) + 10

ymin = min([df_real['LV_Vol_mL'].min(), df_nf['LV_Vol_mL'].min(), df_vae2['LV_Vol_mL'].min(), df_vae3['LV_Vol_mL'].min()]) - 10
ymax = max([df_real['LV_Vol_mL'].max(), df_nf['LV_Vol_mL'].max(), df_vae2['LV_Vol_mL'].max(), df_vae3['LV_Vol_mL'].max()]) + 10

for i, (case, conf_str) in enumerate(zip(diff_plots_real, diff_plots_str)):

    if i == 0:
        case1_color = 'teal'
        case2_color = 'purple'
    elif i == 1:
        case1_color = 'darkgreen'
        case2_color = 'olive'
    else:
        case1_color = "peru"
        case2_color = "brown"

    generate_plot(df=df_real, metadata_case=case, model_str='real', metadata_str=conf_str,
                  xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, case1_color=case1_color, case2_color=case2_color)


### we do the same thing for the most performant generative models
models_str = ['nf', 'vae2', 'vae3']
dfs_gen = [df_nf, df_vae2, df_vae3]

generated_metadata_female = np.load('../data_models_saved/data/metadata_sampled_female.npy')
generated_metadata_male = np.load('../data_models_saved/data/metadata_sampled_male.npy')
generated_metadata = np.concatenate((generated_metadata_female, generated_metadata_male), axis=0)

for mod_str, df in zip(models_str, dfs_gen):

    ### indices according to sex
    female_ind = np.arange(0, 300, 1, dtype=int)
    male_ind = np.arange(300, 600, 1, dtype=int)

    ### indices according to bmi
    small_bmi = np.where(generated_metadata[:, 0] <= 22)[0]
    large_bmi = np.where(generated_metadata[:, 0] > 22)[0]

    ### indices according to age
    small_age = np.where(generated_metadata[:, 1] <= 61)[0]
    large_age = np.where(generated_metadata[:, 1] > 61)[0]

    diff_plots = [[female_ind, male_ind], [small_bmi, large_bmi], [small_age, large_age]]


    for i, (case, conf_str) in enumerate(zip(diff_plots, diff_plots_str)):

        if i == 0:
            case1_color = 'teal'
            case2_color = 'purple'
        elif i == 1:
            case1_color = 'darkgreen'
            case2_color = 'olive'
        else:
            case1_color = "peru"
            case2_color = "brown"

        generate_plot(df=df, metadata_case=case, model_str=mod_str, metadata_str=conf_str,
                      xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, case1_color=case1_color, case2_color=case2_color)




