import pyvista as pv
import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
from image_utils import *
import torch
import networks
from DegradProcess import batch_degrade, MotionDegrad
from torch import optim
import torch.nn.functional as F
import torch
import butils as bu
from typing import Union, Any

gpu_id = 0
device = torch.device("cuda:{:d}".format(
    gpu_id) if torch.cuda.is_available() else "cpu")

#|%%--%%| <jPf9AjtHCm|aC7MIQC2gf>
r"""°°°
### Load UKBB SA data
°°°"""
# |%%--%%| <aC7MIQC2gf|FsaugFONWt>


def get_data(dirs: [Any]) -> torch.Tensor:
    """Get data

    :param dirs: [TODO:description]
    :return: [TODO:return]
    """
    batched_data = []

    for d in dirs:
        seg_data = nib.load(f"./Dataset/{d}/LR_ED.nii.gz").get_fdata()
        #################################################################
        ####### TODO NB still need to align and crop volumes ############
        #################################################################
        one_hotted = torch.Tensor(label2onehot(
            seg_data).transpose(0, 3, 2, 1)[np.newaxis, :])
        batched_data.append(one_hotted)

    return batched_data

#|%%--%%| <FsaugFONWt|lygWvxmbvi>

data_directories = list(range(5))
data = get_data(data_directories) 

# |%%--%%| <lygWvxmbvi|vBLAQMwngD>
r"""°°°
### Load generative models of HR segmentation
°°°"""
# |%%--%%| <vBLAQMwngD|Lx9PmqZMXK>

# load model
z_dim, beta = 64, 1e-3
model = networks.GenVAE3D(z_dim=z_dim, img_size=128, depth=64)
model.to(device)
model_path = 'models/betaVAE/VAECE_zdim_{:d}_epoch_100_beta_{:.2E}_alpha.pt'.format(
    z_dim, beta)
model.load_state_dict(torch.load(model_path, map_location=device))

# |%%--%%| <Lx9PmqZMXK|ogXOAsr6Ji>
r"""°°°
### Enhance the segmentation by joint motion correction and super resolution
°°°"""
# |%%--%%| <ogXOAsr6Ji|A9iAmvJ47h>


def get_z(data_not_batched: torch.Tensor,
          model: torch.nn.Module = model,
          iterations: int = 500
          ):
    MotionLayer = MotionDegrad(
        newD=data_not_batched.shape[-3], newH=data_not_batched.shape[-2], newW=data_not_batched.shape[-1], mode='bilinear').to(device)
    MotionLayer.to(device)

    # LATENT OPTIMISATION
    z0 = torch.zeros((1, z_dim)).to(device)
    seg_map = torch.argmax(data_not_batched, axis=1)

    z_recall = z0.clone().detach().requires_grad_(True)

    # optimizer for z
    optimizer1 = optim.Adam([{'params': z_recall}], lr=0.2)
    scheduler1 = optim.lr_scheduler.StepLR(
        optimizer1, step_size=100, gamma=0.5)
    # optimizer for motion
    optimizer2 = optim.Adam(MotionLayer.parameters(), lr=0.1)
    scheduler2 = optim.lr_scheduler.StepLR(
        optimizer2, step_size=100, gamma=0.5)

    for k in tqdm(range(iterations)):
        # E - step, estimate motion
        optimizer2.zero_grad()
        recon_x = model.decode(z_recall)
        recon_x = MotionLayer(recon_x)
        loss = F.cross_entropy(recon_x, seg_map, reduction='mean')
        loss.backward()
        optimizer2.step()

        # M - step, estimate z
        optimizer1.zero_grad()
        recon_x = model.decode(z_recall)
        recon_x = MotionLayer(recon_x)
        loss = F.cross_entropy(recon_x, seg_map, reduction='mean')
        loss.backward()
        optimizer1.step()

        scheduler1.step()
        scheduler2.step()

    return z_recall

# |%%--%%| <A9iAmvJ47h|omExjGkMwd>



#|%%--%%| <omExjGkMwd|MsmCrWYS8r>

z_vals = []
for d in data:
    z_vals.append(get_z(d, model, iterations=500))

#|%%--%%| <MsmCrWYS8r|tI3bUDcDKj>

z_vals

# |%%--%%| <tI3bUDcDKj|q9uZAAvrFb>

ex_z.shape

# |%%--%%| <q9uZAAvrFb|7FJvFUjeOD>


# |%%--%%| <7FJvFUjeOD|LHHICSZ2NR>

SR_data = onehot2label(model.decode(ex_z).squeeze().detach().cpu().numpy())

# |%%--%%| <LHHICSZ2NR|lcCYYq5FiE>

SR_data.shape

# |%%--%%| <lcCYYq5FiE|VHyIDnSvGU>

vol3view(SR_data)

# |%%--%%| <VHyIDnSvGU|0UJX7mAg3A>

recon = onehot2label(data[1].squeeze().detach().cpu().numpy())

# |%%--%%| <0UJX7mAg3A|aTQvohXDRY>


# |%%--%%| <aTQvohXDRY|2nRwpfBBz

# |%%--%%| <|Qyr9hUlCbl>

# vol3view(SR_data)
vol3view(onehot2label(
    data[1].squeeze().detach().cpu().numpy()).transpose(2, 1, 0))
