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
import matplotlib.pyplot as plt
from pathlib import Path

gpu_id = 0
device = torch.device("cuda:{:d}".format(
    gpu_id) if torch.cuda.is_available() else "cpu")
iterations = 800

#|%%--%%| <dgWF56yuCp|aC7MIQC2gf>
r"""°°°
### Load UKBB SA data
°°°"""
# |%%--%%| <aC7MIQC2gf|i4fYtLfl71>


def coord_mat(shape: [int]):
    x, y, z = np.meshgrid(*[np.arange(i) for i in shape], indexing='ij')
    return np.stack((x, y, z), axis=3)


def get_centre_of_mass(seg_data: np.array):

    mask = seg_data == 0
    bin = np.ones(seg_data.shape)
    bin[mask] = 0

    coords = coord_mat(seg_data.shape)
    return (np.sum(coords * bin[:, :, :, np.newaxis], axis=(0, 1, 2)) / np.sum(bin)).astype(int)


def process_ukb_data(data):

    data = align_UKBvolume(data)
    data = data[::-1, ::-1, :]
    center_of_mass = get_centre_of_mass(data)
    center = center_of_mass.copy()
    center[0] = data.shape[0]//2
    size = [88]*3
    size[0] = data.shape[0]
    return crop_3Dimage(data, center, size)


#|%%--%%| <i4fYtLfl71|bnuNcyzXzk>

def extract_latent(path_to_mri: str):
    mri_dir: str = "/".join(path_to_mri.split("/")[:-1])
    processed = process_ukb_data(nib.load(path_to_mri).get_fdata())
    seg_data = torch.Tensor(label2onehot(processed)[np.newaxis, :])

    # load model
    z_dim, beta = 64, 1e-3
    model = networks.GenVAE3D(z_dim=z_dim, img_size=128, depth=64)
    model.to(device)
    model_path = 'models/betaVAE/VAECE_zdim_{:d}_epoch_100_beta_{:.2E}_alpha.pt'.format(
        z_dim, beta)
    model.load_state_dict(torch.load(model_path, map_location=device))

    MotionLayer = MotionDegrad(
        newD=seg_data.shape[-3], newH=seg_data.shape[-2], newW=seg_data.shape[-1], mode='bilinear').to(device)
    MotionLayer.to(device)

    # LATENT OPTIMISATION
    z0 = torch.zeros((1, z_dim)).to(device)
    seg_map = torch.argmax(seg_data, axis=1)

    z_recall = z0.clone().detach().requires_grad_(True)

    # optimizer for z
    optimizer1 = optim.Adam([{'params': z_recall}], lr=0.2)
    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=100, gamma=0.5)
    # optimizer for motion
    optimizer2 = optim.Adam(MotionLayer.parameters(), lr=0.1)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=100, gamma=0.5)

    for k in tqdm(range(0, iterations)):
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

    z_np = z_recall.detach().cpu().numpy()
    np.savetxt(f"{mri_dir}/latent.csv", z_np, delimiter=',')


#|%%--%%| <bnuNcyzXzk|VdiQJAE2JH>

for sub_dir_name in tqdm(range(1331)):
    try:
        dir_name = f"./Dataset/{sub_dir_name}"
        if not Path(f"{dir_name}/latent.csv").exists():
            mri_pth = f"{dir_name}/LR_ED.nii.gz"
            extract_latent(mri_pth)

    except:
        pass



