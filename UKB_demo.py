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
import matplotlib.pyplot as plt

gpu_id = 0
device = torch.device("cuda:{:d}".format(gpu_id) if torch.cuda.is_available() else "cpu")

# |%%--%%| <bAioCjmYVP|aC7MIQC2gf>
r"""°°°
### Load UKBB SA data
°°°"""
# |%%--%%| <aC7MIQC2gf|FsaugFONWt>


data_dir = './data/UKB_demo'
seg_file_path = os.path.join(data_dir, 'demo_crop.nii.gz')
seg_nib = nib.load(seg_file_path)
seg_data = seg_nib.get_fdata()
seg_LR = torch.Tensor(label2onehot(seg_data)[np.newaxis,:]).to(device)

#|%%--%%| <FsaugFONWt|Usmvdw3byI>
seg_LR.shape


#|%%--%%| <Usmvdw3byI|Zx5sVA1fXb>

for i in range(seg_LR.shape[2]):
    plt.figure()
    plt.imshow(seg_LR[0, 3, i, :, :].cpu().numpy())

# |%%--%%| <Zx5sVA1fXb|vBLAQMwngD>
r"""°°°
### Load generative models of HR segmentation
°°°"""
# |%%--%%| <vBLAQMwngD|Lx9PmqZMXK>

# load model
z_dim, beta = 64, 1e-3
model = networks.GenVAE3D(z_dim=z_dim, img_size=128, depth=64)
model.to(device)
model_path = 'models/betaVAE/VAECE_zdim_{:d}_epoch_100_beta_{:.2E}_alpha.pt'.format(z_dim, beta)
model.load_state_dict(torch.load(model_path, map_location=device))

# |%%--%%| <Lx9PmqZMXK|ogXOAsr6Ji>
r"""°°°
### Enhance the segmentation by joint motion correction and super resolution
°°°"""
# |%%--%%| <ogXOAsr6Ji|uwxUUeBHuK>

MotionLayer = MotionDegrad(newD=seg_LR.shape[-3], newH=seg_LR.shape[-2], newW=seg_LR.shape[-1], mode='bilinear').to(device)
MotionLayer.to(device)

# LATENT OPTIMISATION
z0 = torch.zeros((1, z_dim)).to(device)
seg_map = torch.argmax(seg_LR, axis=1)

z_recall = z0.clone().detach().requires_grad_(True)

# optimizer for z
optimizer1 = optim.Adam([{'params': z_recall}], lr=0.2)
scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=100, gamma=0.5)
# optimizer for motion
optimizer2 = optim.Adam(MotionLayer.parameters(), lr=0.1)
scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=100, gamma=0.5)

for k in tqdm(range(0, 500)):
    # E - step, estimate motion
    optimizer2.zero_grad()
    recon_x = model.decode(z_recall)
    recon_x = MotionLayer(recon_x)
    loss = F.cross_entropy(recon_x, seg_map,reduction='mean')
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

#|%%--%%| <uwxUUeBHuK|omExjGkMwd>


vol3view(SR_data)
#|%%--%%| <omExjGkMwd|O1Vo5ycozF>



