import pickle

with open("../ablation/kl_dict_out_dim_conf.pkl", "rb") as f:
    my_dict = pickle.load(f)

print(my_dict['Myo_Mass_g'])