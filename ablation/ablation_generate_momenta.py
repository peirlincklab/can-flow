import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
from utils import save_write_momenta


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


scaler = MinMaxScaler()


#x_confounders = pd.read_excel(r"/home/kevopou1/metadata_final.xlsx")
x_confounders = pd.read_excel(r"C:\Users\kkevopoulos\OneDrive - Delft University of Technology\Bureaublad\data_kostas_bivme\metadata_final.xlsx")
x_confounders.drop(['Participant ID', 'Height', 'Weight', 'Diastolic BP',
                    'Systolic BP', 'Unnamed: 8', 'Unnamed: 9', 'subject_id'], axis=1, inplace=True)
x_confounders = pd.get_dummies(x_confounders, columns=['Sex'])
x_confounders['Sex_Female'] = x_confounders['Sex_Female'].replace({True: 1, False: 0})
x_confounders['Sex_Male'] = x_confounders['Sex_Male'].replace({True: 1, False: 0})
x_confounders = x_confounders[['BMI', 'Age', 'Sex_Female', 'Sex_Male']]
x_confounders = x_confounders.to_numpy()

NumAll = 2274
frac_train = 0.7
NumTrainSamples = int(NumAll * frac_train)

X_train_confounders = scaler.fit_transform(x_confounders[:NumTrainSamples, :])


metadata_sampled_female = np.load('../data_models_saved/data/metadata_sampled_female.npy')
metadata_sampled_male = np.load('../data_models_saved/data/metadata_sampled_male.npy')

metadata_sampled_all = np.concatenate((metadata_sampled_female, metadata_sampled_male), axis=0)
metadata_sampled_all = scaler.transform(metadata_sampled_all)
metadata_sampled_all = torch.tensor(metadata_sampled_all, dtype=torch.float32, device=device)

ae_decoder = torch.load("../data_models_saved/models/ae_decoder.pth", weights_only=False)


folder_path = "../ablation/models"  # relative to your project root

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    model_abl = torch.load(file_path, weights_only=False)

    z_synthetic, _ = model_abl.reverse(metadata_sampled_all)
    generated_shapes = ae_decoder(z_synthetic)

    generated_shapes = generated_shapes.reshape(metadata_sampled_all.shape[0], 720, 3)
    generated_shapes = generated_shapes.detach().cpu().numpy()

    filename_save = os.path.splitext(filename)[0]

    save_write_momenta.save_momenta(type_momenta=filename_save, momenta_tosave=generated_shapes)

