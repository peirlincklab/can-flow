import numpy as np
import os


def sample_metadata_targeted(num_samples, gen_sex, gen_bmi, gen_age):
    sex1_all = np.tile(gen_sex[0], num_samples).reshape(-1, 1)
    sex2_all = np.tile(gen_sex[1], num_samples).reshape(-1, 1)

    min_bmi = gen_bmi[0]
    max_bmi = gen_bmi[1]

    min_age = gen_age[0]
    max_age = gen_age[1]

    gen_sex_samples = np.concatenate((sex1_all, sex2_all), axis=1)

    gen_bmi_samples = np.random.uniform(low=min_bmi, high=max_bmi, size=num_samples).reshape(-1, 1)
    gen_age_samples = np.random.uniform(low=min_age, high=max_age, size=num_samples).reshape(-1, 1)

    metadata_samples = np.concatenate((gen_bmi_samples, gen_age_samples, gen_sex_samples), axis=1)

    return metadata_samples

os.makedirs('../data_models_saved/data/metadata_sampled_synthetic_example_bins', exist_ok=True)

sex_female = [1, 0]
sex_male = [0, 1]

age1 = [50, 60]
age2 = [60, 70]
age3 = [70, 80]

bmi1 = [17, 21]
bmi2 = [21, 25]
bmi3 = [25, 29]

sex_vals = [sex_female, sex_male]
age_vals = [age1, age2, age3]
bmi_vals = [bmi1, bmi2, bmi3]

samples_per_bin = 20

for sex in sex_vals:
    for age in age_vals:
        for bmi in bmi_vals:

            sampled_metadata_bin = sample_metadata_targeted(num_samples=samples_per_bin,
                                                        gen_sex=sex, gen_bmi=bmi, gen_age=age)

            sex_str = 'female' if sex == [1, 0] else 'male'

            np.save(f'../data_models_saved/data/metadata_sampled_synthetic_example_bins/{sex_str}_{age}_{bmi}.npy', sampled_metadata_bin)

