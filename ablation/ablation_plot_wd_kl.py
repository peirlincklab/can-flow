import pickle

with open("../ablation/wass_dict_n_flow.pkl", "rb") as f:
    my_dict = pickle.load(f)

print(my_dict['LV_Vol_mL'])