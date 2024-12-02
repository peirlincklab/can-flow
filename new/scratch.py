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

gpu_id = 0
device = torch.device("cuda:{:d}".format(
    gpu_id) if torch.cuda.is_available() else "cpu")
iterations = 800

#|%%--%%| <dK2EQH3Lyk|aC7MIQC2gf>
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


# |%%--%%| <i4fYtLfl71|61jzgkb8qI>


processed = process_ukb_data(
    nib.load(f"./Dataset/100/LR_ED.nii.gz").get_fdata())
processed.shape

# |%%--%%| <61jzgkb8qI|OfvxFLoPGV>

seg_data = torch.Tensor(label2onehot(processed)[np.newaxis, :])

# |%%--%%| <OfvxFLoPGV|NupsZkixVU>

seg_data.shape

# |%%--%%| <NupsZkixVU|8B9Z5ulWMV>

data_dir = './data/UKB_demo'
seg_file_path = os.path.join(data_dir, 'demo_crop.nii.gz')
seg_nib = nib.load(seg_file_path)
seg_data_demo = seg_nib.get_fdata()
seg_LR = torch.Tensor(label2onehot(seg_data_demo)[np.newaxis, :]).to(device)


# |%%--%%| <8B9Z5ulWMV|9vABWmilmM>


for i in range(seg_data.shape[2]):
    plt.figure()
    plt.title("Mine")
    plt.imshow(seg_data[0, 3, i, :, :].cpu().numpy())
    plt.figure()
    plt.title("demo")
    plt.imshow(seg_LR[0, 3, i, :, :].cpu().numpy())


# |%%--%%| <9vABWmilmM|RAc7kaLIfz>

for i in range(0, seg_data.shape[3], 10):
    plt.figure()
    plt.title("Mine")
    plt.imshow(seg_data[0, 0, :,  i, :])
    plt.figure()
    plt.title("Demo")
    plt.imshow(seg_LR[0, 0, :,   i, :])

# |%%--%%| <RAc7kaLIfz|2kzmsseGSH>


# load model
z_dim, beta = 64, 1e-3
model = networks.GenVAE3D(z_dim=z_dim, img_size=128, depth=64)
model.to(device)
model_path = 'models/betaVAE/VAECE_zdim_{:d}_epoch_100_beta_{:.2E}_alpha.pt'.format(
    z_dim, beta)
model.load_state_dict(torch.load(model_path, map_location=device))

# |%%--%%| <2kzmsseGSH|miLKu2EDps>
r"""°°°
### Enhance the segmentation by joint motion correction and super resolution
°°°"""
# |%%--%%| <miLKu2EDps|uwxUUeBHuK>

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

SR_data = onehot2label(model.decode(z_recall).squeeze().detach().cpu().numpy())

# |%%--%%| <uwxUUeBHuK|FSHVYyAo8S>


vol3view(SR_data)

# |%%--%%| <FSHVYyAo8S|O1Vo5ycozF>


# # |%%--%%| <O1Vo5ycozF|AW7R1k6XUm>

# # load model
# z_dim, beta = 64, 1e-3
# model = networks.GenVAE3D(z_dim=z_dim, img_size=128, depth=64)
# model.to(device)
# model_path = 'models/betaVAE/VAECE_zdim_{:d}_epoch_100_beta_{:.2E}_alpha.pt'.format(
#     z_dim, beta)
# model.load_state_dict(torch.load(model_path, map_location=device))


# # |%%--%%| <AW7R1k6XUm|17uNTxi6UE>


# MotionLayer = MotionDegrad(
#     newD=seg_data.shape[-3], newH=seg_data.shape[-2], newW=seg_data.shape[-1], mode='bilinear').to(device)
# MotionLayer.to(device)

# # LATENT OPTIMISATION
# z0 = torch.zeros((1, z_dim)).to(device)
# seg_map = torch.argmax(seg_data, axis=1)

# z_recall = z0.clone().detach().requires_grad_(True)

# # optimizer for z
# optimizer1 = optim.Adam([{'params': z_recall}], lr=0.2)
# scheduler1 = optim.lr_scheduler.StepLR(
#     optimizer1, step_size=100, gamma=0.5)
# # optimizer for motion
# optimizer2 = optim.Adam(MotionLayer.parameters(), lr=0.1)
# scheduler2 = optim.lr_scheduler.StepLR(
#     optimizer2, step_size=100, gamma=0.5)

# for k in tqdm(range(iterations)):
#     # E - step, estimate motion
#     optimizer2.zero_grad()
#     recon_x = model.decode(z_recall)
#     recon_x = MotionLayer(recon_x)
#     loss = F.cross_entropy(recon_x, seg_map, reduction='mean')
#     loss.backward()
#     optimizer2.step()

#     # M - step, estimate z
#     optimizer1.zero_grad()
#     recon_x = model.decode(z_recall)
#     recon_x = MotionLayer(recon_x)
#     loss = F.cross_entropy(recon_x, seg_map, reduction='mean')
#     loss.backward()
#     optimizer1.step()

#     scheduler1.step()
#     scheduler2.step()


# # |%%--%%| <17uNTxi6UE|nNTwojjovk>

# SR_data = onehot2label(model.decode(z_recall).squeeze().detach().cpu().numpy())

# # |%%--%%| <nNTwojjovk|MkeuKfHxRW>

# SR_data.shape

# # |%%--%%| <MkeuKfHxRW|m6b9PuHAd6>

# vol3view(SR_data)


# # |%%--%%| <m6b9PuHAd6|iyuLKq7qxe>

# # vol3view(seg_data)
# vol3view(cropped)

# # |%%--%%| <iyuLKq7qxe|QKy4mjxAVo>

# aligned.shape


# # |%%--%%| <QKy4mjxAVo|FsaugFONWt>


# def calculate_and_save_latent(path_to_mri: str) -> None:

#     batched_data = []

#     for d in dirs:
#         seg_data = nib.load(f"./Dataset/{d}/LR_ED.nii.gz").get_fdata()
#         #################################################################
#         ####### TODO NB still need to align and crop volumes ############
#         #################################################################
#         one_hotted = torch.Tensor(label2onehot(
#             seg_data).transpose(0, 3, 2, 1)[np.newaxis, :])
#         batched_data.append(one_hotted)

#     return batched_data

# # |%%--%%| <FsaugFONWt|lygWvxmbvi>


# data_directories = list(range(5))
# data = get_data(data_directories)

# # |%%--%%| <lygWvxmbvi|vBLAQMwngD>
# r"""°°°
# ### Load generative models of HR segmentation
# °°°"""
# # |%%--%%| <vBLAQMwngD|Lx9PmqZMXK>

# # load model
# z_dim, beta = 64, 1e-3
# model = networks.GenVAE3D(z_dim=z_dim, img_size=128, depth=64)
# model.to(device)
# model_path = 'models/betaVAE/VAECE_zdim_{:d}_epoch_100_beta_{:.2E}_alpha.pt'.format(
#     z_dim, beta)
# model.load_state_dict(torch.load(model_path, map_location=device))

# # |%%--%%| <Lx9PmqZMXK|ogXOAsr6Ji>
# r"""°°°
# ### Enhance the segmentation by joint motion correction and super resolution
# °°°"""
# # |%%--%%| <ogXOAsr6Ji|A9iAmvJ47h>


# def get_z(data_not_batched: torch.Tensor,
#           model: torch.nn.Module = model,
#           iterations: int = 500
#           ):

#     MotionLayer = MotionDegrad(
#         newD=data_not_batched.shape[-3], newH=data_not_batched.shape[-2], newW=data_not_batched.shape[-1], mode='bilinear').to(device)
#     MotionLayer.to(device)

#     # LATENT OPTIMISATION
#     z0 = torch.zeros((1, z_dim)).to(device)
#     seg_map = torch.argmax(data_not_batched, axis=1)

#     z_recall = z0.clone().detach().requires_grad_(True)

#     # optimizer for z
#     optimizer1 = optim.Adam([{'params': z_recall}], lr=0.2)
#     scheduler1 = optim.lr_scheduler.StepLR(
#         optimizer1, step_size=100, gamma=0.5)
#     # optimizer for motion
#     optimizer2 = optim.Adam(MotionLayer.parameters(), lr=0.1)
#     scheduler2 = optim.lr_scheduler.StepLR(
#         optimizer2, step_size=100, gamma=0.5)

#     for k in tqdm(range(iterations)):
#         # E - step, estimate motion
#         optimizer2.zero_grad()
#         recon_x = model.decode(z_recall)
#         recon_x = MotionLayer(recon_x)
#         loss = F.cross_entropy(recon_x, seg_map, reduction='mean')
#         loss.backward()
#         optimizer2.step()

#         # M - step, estimate z
#         optimizer1.zero_grad()
#         recon_x = model.decode(z_recall)
#         recon_x = MotionLayer(recon_x)
#         loss = F.cross_entropy(recon_x, seg_map, reduction='mean')
#         loss.backward()
#         optimizer1.step()

#         scheduler1.step()
#         scheduler2.step()

#     return z_recall

# # |%%--%%| <A9iAmvJ47h|omExjGkMwd>


# # |%%--%%| <omExjGkMwd|MsmCrWYS8r>
# z_vals = []
# for d in data:
#     z_vals.append(get_z(d, model, iterations=500))

# # |%%--%%| <MsmCrWYS8r|tI3bUDcDKj>

# z_vals

# # |%%--%%| <tI3bUDcDKj|q9uZAAvrFb>

# ex_z.shape

# # |%%--%%| <q9uZAAvrFb|7FJvFUjeOD>


# # |%%--%%| <7FJvFUjeOD|LHHICSZ2NR>

# SR_data = onehot2label(model.decode(ex_z).squeeze().detach().cpu().numpy())

# # |%%--%%| <LHHICSZ2NR|lcCYYq5FiE>

# SR_data.shape

# # |%%--%%| <lcCYYq5FiE|VHyIDnSvGU>

# vol3view(SR_data)

# # |%%--%%| <VHyIDnSvGU|0UJX7mAg3A>

# recon = onehot2label(data[1].squeeze().detach().cpu().numpy())

# # |%%--%%| <0UJX7mAg3A|aTQvohXDRY>


# # |%%--%%| <aTQvohXDRY|2nRwpfBBz

# # |%%--%%| <|Qyr9hUlCbl>

# # vol3view(SR_data)
# vol3view(onehot2label(
#     data[1].squeeze().detach().cpu().numpy()).transpose(2, 1, 0))
