import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from utils import save_write_momenta

np.random.seed(50)


scaler = MinMaxScaler()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

frac_train = 0.7
frac_valid = 0.15


NumAll =  2274
NumTrainSamples = int(NumAll * frac_train)
NumValidSamples = int(NumAll * frac_valid)

### Load the confounders just to apply the scaler to the training data
### And also to compute the boundaries of the metadata parameter space
# x_confounders = pd.read_excel(r"/home/kevopou1/metadata_final.xlsx")
x_confounders = pd.read_excel(r"C:\Users\kkevopoulos\OneDrive - Delft University of Technology\Bureaublad\data_kostas_bivme\metadata_final.xlsx")
x_confounders.drop(['Participant ID', 'Height', 'Weight', 'Diastolic BP',
                    'Systolic BP', 'Unnamed: 8', 'Unnamed: 9', 'subject_id'], axis=1, inplace=True)
x_confounders = pd.get_dummies(x_confounders, columns=['Sex'])
x_confounders['Sex_Female'] = x_confounders['Sex_Female'].replace({True: 1, False: 0})
x_confounders['Sex_Male'] = x_confounders['Sex_Male'].replace({True: 1, False: 0})
x_confounders = x_confounders[['BMI', 'Age', 'Sex_Female', 'Sex_Male']]
x_confounders = x_confounders.to_numpy()

x_confounders_train = scaler.fit_transform(x_confounders[:NumTrainSamples, :])
x_confounders_valid = scaler.transform(x_confounders[NumTrainSamples:NumTrainSamples+NumValidSamples])
x_confounders_test = scaler.transform(x_confounders[NumTrainSamples+NumValidSamples:])

x_confounders_test = torch.tensor(x_confounders_test, dtype=torch.float32, device=device)


ae_decoder = torch.load("../data_models_saved/models/ae_decoder.pth", weights_only=False)
cnf = torch.load("../data_models_saved/models/cnf_model.pth", weights_only=False)

num_gen = 70
x_conf_to_gen = x_confounders_test[-1].repeat((num_gen, 1))
### Choose an individual randomly
z_synthetic, _ = cnf.reverse(x_conf_to_gen)

generated_shapes = ae_decoder(z_synthetic)

generated_shapes = generated_shapes.reshape(num_gen, 720, 3)
generated_shapes = generated_shapes.detach().cpu().numpy()

save_write_momenta.save_momenta(type_momenta="Specific_individual", momenta_tosave=generated_shapes)
print(x_confounders[-1])


