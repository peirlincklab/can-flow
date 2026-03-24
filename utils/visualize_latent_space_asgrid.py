import numpy as np
import torch
import matplotlib.pyplot as plt

X_train_z = np.load("../data_models_saved/data/X_train_z.npy")

latent_dim = X_train_z.shape[1]

### CAN-DO AE latent sample
z_latent_cando = X_train_z[0].reshape(1, 44)
z_latent_cando =z_latent_cando.reshape(4, 11)



latent_dist = torch.distributions.MultivariateNormal(torch.zeros(latent_dim), torch.eye(latent_dim))
z_latent_normal = latent_dist.sample((1,))
z_latent_normal = z_latent_normal.detach().numpy()
z_latent_normal = z_latent_normal.reshape(4, 11)

vmin = min(z_latent_cando.min(), z_latent_normal.min())
vmax = max(z_latent_cando.max(), z_latent_normal.max())


plt.figure()
plt.imshow(z_latent_cando, cmap='plasma', vmin=vmin, vmax=vmax)
plt.xticks([])
plt.yticks([])
plt.box(False)

plt.savefig('cando_ae_latent.svg')

plt.figure()
plt.imshow(z_latent_normal, cmap='plasma', vmin=vmin, vmax=vmax)
plt.xticks([])
plt.yticks([])
plt.box(False)
plt.savefig('normal_latent.svg')