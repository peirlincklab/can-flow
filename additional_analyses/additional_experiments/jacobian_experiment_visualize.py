import numpy as np
import matplotlib.pyplot as plt

# 7 categories
labels = [
    "CAN-DO",
    r"cVAE $\beta=10^{-1}$",
    r"cVAE $\beta=10^{-2}$",
    r"cVAE $\beta=10^{-3}$",
    r"cVAE $\beta=10^{-4}$",
    r"cVAE $\beta=10^{-5}$",
    r"cVAE $\beta=10^{-6}$"
]

# values for one method
mean = np.load('../data_results/eranks_all_mean.npy')
std = np.load('../data_results/eranks_all_std.npy')
upper = mean + std
lower = mean - std

# close the circle
mean  = np.concatenate([mean,[mean[0]]])
upper = np.concatenate([upper,[upper[0]]])
lower = np.concatenate([lower,[lower[0]]])

angles = np.linspace(0,2*np.pi,len(labels),endpoint=False)
angles = np.concatenate([angles,[angles[0]]])

fig, ax = plt.subplots(figsize=(6,6),subplot_kw=dict(polar=True))

ax.plot(angles, mean, linewidth=2, label="Mean")

ax.fill_between(
    angles,
    lower,
    upper,
    alpha=0.2,
    label="± std"
)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

ax.legend()
plt.show()