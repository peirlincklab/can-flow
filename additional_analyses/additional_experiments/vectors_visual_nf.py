import numpy as np
import torch
import pyvista as pv
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

scaler = MinMaxScaler()

frac_train = 0.7

NumAll =  2274
NumTrainSamples = int(NumAll * frac_train)

x_confounders = pd.read_excel(r"C:\Users\kkevopoulos\OneDrive - Delft University of Technology\Bureaublad\data_kostas_bivme\metadata_final.xlsx")
x_confounders.drop(['Participant ID', 'Height', 'Weight', 'Diastolic BP',
                    'Systolic BP', 'Unnamed: 8', 'Unnamed: 9', 'subject_id'], axis=1, inplace=True)
x_confounders = pd.get_dummies(x_confounders, columns=['Sex'])
x_confounders['Sex_Female'] = x_confounders['Sex_Female'].replace({True: 1, False: 0})
x_confounders['Sex_Male'] = x_confounders['Sex_Male'].replace({True: 1, False: 0})
x_confounders = x_confounders[['BMI', 'Age', 'Sex_Female', 'Sex_Male']]
x_confounders = x_confounders.to_numpy()

x_confounders_train = scaler.fit_transform(x_confounders[:NumTrainSamples, :])

x_conf = x_confounders_train[4, None]
x_conf = torch.tensor(x_conf, dtype=torch.float32, device=device)


momenta = np.loadtxt("../../data_models_saved/data/DeterministicAtlas__EstimatedParameters__Momenta.txt")

momenta = np.delete(momenta, 0, axis=0)
momenta = momenta.reshape((2274, 720, 3))

x_momenta = momenta[4, :, :]

cps = np.loadtxt("../../data_models_saved/data/DeterministicAtlas__EstimatedParameters__ControlPoints.txt")\

ae_decoder = torch.load("../../data_models_saved/models/ae_decoder.pth", weights_only=False)
cnf = torch.load("../../data_models_saved/models/cnf_model.pth", weights_only=False)

ae_decoder.eval()
cnf.eval()

_, outs = cnf.reverse(x_conf)


gen_momenta_random0 = ae_decoder(outs[0])
gen_momenta_random0 = gen_momenta_random0.reshape(1, 720, 3)
gen_momenta_random0 = gen_momenta_random0.squeeze(0)
gen_momenta_random0 = gen_momenta_random0.detach().cpu().numpy()

gen_momenta_random12 = ae_decoder(outs[12])
gen_momenta_random12 = gen_momenta_random12.reshape(1, 720, 3)
gen_momenta_random12 = gen_momenta_random12.squeeze(0)
gen_momenta_random12 = gen_momenta_random12.detach().cpu().numpy()

gen_momenta_random_last = ae_decoder(outs[-1])
gen_momenta_random_last = gen_momenta_random_last.reshape(1, 720, 3)
gen_momenta_random_last = gen_momenta_random_last.squeeze(0)
gen_momenta_random_last = gen_momenta_random_last.detach().cpu().numpy()

cloud = pv.PolyData(cps)
cloud.point_data["random0"] = gen_momenta_random0
cloud.point_data["random12"] = gen_momenta_random12
cloud.point_data["randomlast"] = gen_momenta_random_last
cloud.point_data["real_momenta"] = x_momenta

cloud.save(r"C:\Users\kkevopoulos\Documents\Meshes_Anatomies\vectors_visual.vtk")





