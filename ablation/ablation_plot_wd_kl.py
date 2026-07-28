import pickle

with open("../ablation/wass_dict_out_dim_conf.pkl", "rb") as f:
    my_dict = pickle.load(f)

print(my_dict['LV_Sphericity'])