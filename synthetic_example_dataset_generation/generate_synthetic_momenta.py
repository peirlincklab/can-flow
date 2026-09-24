import numpy as np
from sklearn.preprocessing import  MinMaxScaler
import torch
import pandas as pd
from utils import save_write_momenta
import os

scaler = MinMaxScaler()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


num_before_outliers = 2272
outliers = np.load('../utils/outliers_indices.npy')
mask = np.ones(num_before_outliers, dtype=bool)
mask[outliers] = False

### Load the metadata just to apply the scaler to the training data
### And also to compute the boundaries of the metadata parameter space
x_confounders = pd.read_excel(r"C:\Users\kkevopoulos\OneDrive - Delft University of Technology\Bureaublad\data_kostas_bivme\metadata_final.xlsx")
x_confounders.drop(['Participant ID', 'Height', 'Weight', 'Diastolic BP',
                    'Systolic BP', 'Unnamed: 8', 'Unnamed: 9', 'subject_id'], axis=1, inplace=True)
x_confounders = pd.get_dummies(x_confounders, columns=['Sex'])
x_confounders['Sex_Female'] = x_confounders['Sex_Female'].replace({True: 1, False: 0})
x_confounders['Sex_Male'] = x_confounders['Sex_Male'].replace({True: 1, False: 0})
x_confounders = x_confounders[['BMI', 'Age', 'Sex_Female', 'Sex_Male']]
x_confounders = x_confounders.to_numpy()

### Delete outliers and participants that withdrew from the study
x_confounders = np.delete(x_confounders, [1746, 1831], axis=0)
x_confounders = x_confounders[mask]

frac_train = 0.7
NumAll =  x_confounders.shape[0]
NumTrainSamples = int(NumAll * frac_train)

x_confounders_train = scaler.fit_transform(x_confounders[:NumTrainSamples, :])

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

cnf = torch.load('../data_models_saved/models/cnf_model.pth', weights_only=False)
ae_decoder = torch.load('../data_models_saved/models/ae_decoder.pth', weights_only=False)

cnf.eval()
ae_decoder.eval()

for sex in sex_vals:
    for age in age_vals:
        for bmi in bmi_vals:

            sex_str = 'female' if sex == [1, 0] else 'male'

            metadata = np.load(f'../data_models_saved/data/metadata_sampled_synthetic_example_bins/{sex_str}_{age}_{bmi}.npy')
            metadata = scaler.transform(metadata)
            metadata = torch.tensor(metadata, dtype=torch.float32).to(device)

            z_synthetic, _ = cnf.reverse(metadata)
            momenta_synthetic = ae_decoder(z_synthetic)

            momenta_synthetic = momenta_synthetic.reshape(metadata.shape[0], 720, 3)
            momenta_synthetic = momenta_synthetic.detach().cpu().numpy()

            save_write_momenta.save_momenta(type_momenta=f"{sex_str}_{age}_{bmi}", momenta_tosave=momenta_synthetic)

