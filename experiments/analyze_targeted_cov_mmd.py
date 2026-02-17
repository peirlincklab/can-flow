import numpy as np

mmd_targeted = np.load("../data_models_saved/data/mmd_vals_targeted.npy")
cov_targeted = np.load("../data_models_saved/data/cov_vals_targeted.npy")

mean_mmd = np.mean(mmd_targeted, axis=0)
std_mmd = np.std(mmd_targeted, axis=0)

mean_cov = np.mean(cov_targeted, axis=0)
std_cov = np.std(cov_targeted, axis=0)

print(f"MMD: {mean_mmd} +- {std_mmd}\n"
      f"Cov: {mean_cov} +- {std_cov}")