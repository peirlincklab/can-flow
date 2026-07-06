import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.stats import entropy


### Prepares the data for the spider plot visuals


def find_data_bin(df, pheno, age_min_bin, age_max_bin, metadata):
    """
        Extract phenotype values for samples within a specified age range.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing phenotype values.
        pheno : str
            Name of the phenotype column to extract from `df`.
        age_min_bin : float
            Lower bound of the age bin, inclusive.
        age_max_bin : float
            Upper bound of the age bin, exclusive.
        metadata : np.ndarray
            Metadata array with shape (n_samples, n_metadata_features). The second
            column, `metadata[:, 1]`, is assumed to contain age values.

        Returns
        -------
        pandas.Series
            Phenotype values corresponding to samples whose age lies in the
            specified age bin.
        """
    indices = np.where((metadata[:, 1] >= age_min_bin) & (metadata[:, 1] < age_max_bin))[0]
    pheno_bin = df[pheno][indices]

    return pheno_bin



def kl_div_histogram(x, y, bins=25, eps=1e-12):
    """
        Compute the Kullback-Leibler (KL) divergence between two one-dimensional
        datasets using histogram-based probability estimates.

        First construct a common histogram range based on the minimum
        and maximum values across both input arrays. Then, estimate the empirical
        probability distributions of `x` and `y` using the same bin edges, add a
        small numerical constant to avoid division by zero, and compute the KL
        divergence D_KL(P_x || P_y).

        Parameters
        ----------
        x : array-like
            First input dataset. This distribution is treated as the reference
            distribution P_x in the KL divergence.
        y : array-like
            Second input dataset. This distribution is treated as the comparison
            distribution P_y in the KL divergence.
        bins : int, optional
            Number of histogram bins used to estimate the probability distributions.
            Default is 25.
        eps : float, optional
            Small constant added to the histogram probabilities to avoid numerical
            issues caused by zero probabilities. Default is 1e-12.

        Returns
        -------
        float
            Histogram-based estimate of the KL divergence D_KL(P_x || P_y)
        """
    x = np.asarray(x)
    y = np.asarray(y)

    ### common bin range
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())

    px, bin_edges = np.histogram(x, bins=bins, range=(min_val, max_val), density=False)
    py, _ = np.histogram(y, bins=bin_edges, density=False)

    px = px.astype(float)
    py = py.astype(float)

    px /= px.sum()
    py /= py.sum()

    ### avoid division by zero
    px += eps
    py += eps

    px /= px.sum()
    py /= py.sum()

    return entropy(px, py)


### load metadata file
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

female_ind_real = np.where(x_confounders[:, 2] == 1)[0]
male_ind_real = np.where(x_confounders[:, 2] == 0)[0]

num_gen = 300

### load sampled metadata
metadata_sampled_female = np.load('../data_models_saved/data/metadata_sampled_female.npy')
metadata_sampled_male = np.load('../data_models_saved/data/metadata_sampled_male.npy')

metadata_sampled_all = np.concatenate((metadata_sampled_female, metadata_sampled_male), axis=0)

df_real = pd.read_pickle('../data_models_saved/data/dataframes_clinical_info/df_real_clinical.pkl')


clinical_phenotypes = ['RV_Vol_mL', 'LV_Vol_mL', 'Myo_Mass_g', 'RVEDV_LVEDV_ratio', 'Long_axis_length', 'LV_Sphericity']
models = ['nf', 'vae1', 'vae2', 'vae3', 'vae4', 'vae5', 'vae6']

models_wass_kl = []
for i, model in enumerate(models):
    df_gen = pd.read_pickle(f"../data_models_saved/data/dataframes_clinical_info/df_{model}_clinical.pkl")

    wass = []
    kl = []
    for pheno in clinical_phenotypes:

        ### first, compute p(m|c=sex)

        pheno_female_real = df_real[pheno][female_ind_real]
        pheno_male_real = df_real[pheno][male_ind_real]

        pheno_female_gen = df_gen[pheno][:num_gen]
        pheno_male_gen = df_gen[pheno][num_gen:]


        x_female = np.array(pheno_female_real)
        y_female = np.array(pheno_female_gen)

        x_male = np.array(pheno_male_real)
        y_male = np.array(pheno_male_gen)


        wass_female = wasserstein_distance(x_female, y_female)
        kl_female = kl_div_histogram(x=x_female, y=y_female)

        wass_male = wasserstein_distance(x_male, y_male)
        kl_male = kl_div_histogram(x=x_male, y=y_male)


        ### then, compute p(m|c=age)
        ### split age in 2 bins: [46, 65), [65, 84]
        pheno_bin1_real = find_data_bin(df=df_real, pheno=pheno, age_min_bin=46, age_max_bin=65, metadata=x_confounders)
        pheno_bin2_real = find_data_bin(df=df_real, pheno=pheno, age_min_bin=65, age_max_bin=84, metadata=x_confounders)

        pheno_bin1_gen = find_data_bin(df=df_gen, pheno=pheno, age_min_bin=46, age_max_bin=65, metadata=metadata_sampled_all)
        pheno_bin2_gen = find_data_bin(df=df_gen, pheno=pheno, age_min_bin=65, age_max_bin=84, metadata=metadata_sampled_all)

        x_pheno_bin1 = np.array(pheno_bin1_real)
        x_pheno_bin2 = np.array(pheno_bin2_real)

        y_pheno_bin1 = np.array(pheno_bin1_gen)
        y_pheno_bin2 = np.array(pheno_bin2_gen)


        wass_bin1 = wasserstein_distance(x_pheno_bin1, y_pheno_bin1)
        kl_bin1 = kl_div_histogram(x=x_pheno_bin1, y=y_pheno_bin1)

        wass_bin2 = wasserstein_distance(x_pheno_bin2, y_pheno_bin2)
        kl_bin2 = kl_div_histogram(x=x_pheno_bin2, y=y_pheno_bin2)

        wass.append([[wass_female, wass_male], [wass_bin1, wass_bin2]])
        kl.append([[kl_female, kl_male], [kl_bin1, kl_bin2]])

    models_wass_kl.append([wass, kl])

np.save('../data_models_saved/data/conditional_wd_kl_forspider_2bins.npy',
        np.array(models_wass_kl,dtype=object),
        allow_pickle=True
        )








