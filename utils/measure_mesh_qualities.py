import numpy as np
from tqdm import tqdm
import torch
import pyvista as pv
from torch_cluster import knn



def MMD(S_g, S_r):
    mmds = []

    pbar = tqdm(total=len(S_r), desc='Compute MMD...')
    for ref in S_r:
        cds = [chamfer_distance(ref, gen) for gen in S_g]
        mmds.append(np.min(cds))

        pbar.update()
    pbar.close()

    return np.mean(mmds)


def Coverage(S_g, S_r):

    matched_ref_indices = set()
    pbar = tqdm(total=len(S_g), desc='Compute Coverage...')
    for gen in S_g:
        ### Compute distances from this generated cloud to all reference clouds
        distances = [chamfer_distance(gen, ref) for ref in S_r]
        ### Find the reference cloud that is closest to this generated cloud
        closest_ref_idx = np.argmin(distances)
        matched_ref_indices.add(closest_ref_idx)

        pbar.update()
    pbar.close()

    return len(matched_ref_indices) / len(S_r)


def chamfer_distance(pc1, pc2):
    """
    pc1: (N, 3) CUDA tensor
    pc2: (M, 3) CUDA tensor
    """

    pc1 = torch.tensor(pc1, dtype=torch.float32).to('cuda')
    pc2 = torch.tensor(pc2, dtype=torch.float32).to('cuda')
    ### pc1 → pc2
    idx12 = knn(pc2, pc1, k=1)
    nn12 = pc2[idx12[1]]
    d12 = ((pc1 - nn12) ** 2).sum(dim=1)

    ### pc2 → pc1
    idx21 = knn(pc1, pc2, k=1)
    nn21 = pc1[idx21[1]]
    d21 = ((pc2 - nn21) ** 2).sum(dim=1)

    distance = d12.mean() + d21.mean()

    return distance.detach().cpu().numpy()


def load_vtk_mesh(file_path):
    """Load a surface mesh from a VTK file using PyVista and return its points."""
    mesh = pv.read(file_path)
    return np.array(mesh.points)


def one_nna(S_r, S_g):

    S_r = np.stack(S_r, axis=0)
    S_g = np.stack(S_g, axis=0)

    S_r = torch.tensor(S_r, dtype=torch.float32)
    S_g = torch.tensor(S_g, dtype = torch.float32)

    all_points = torch.cat([S_r, S_g], dim=0)
    labels = torch.cat([
        torch.zeros(len(S_r), dtype=torch.long),
        torch.ones(len(S_g), dtype=torch.long)
    ])

    correct = 0
    N = len(all_points)
    
    pbar = tqdm(total=N, leave=True,  desc='Compute 1-NNA...')
    for i in range(N):
        x = all_points[i]

        mask = torch.arange(N) != i
        others = all_points[mask]
        other_labels = labels[mask]

        distances = []
        for y in others:
            dist = chamfer_distance(x, y)
            distances.append(torch.tensor(dist, dtype=torch.float32))

        distances = torch.stack(distances)
        nn_idx = torch.argmin(distances)

        if labels[i] == other_labels[nn_idx]:
            correct += 1

        print(f'Running 1-NNA: {correct/N}')

        pbar.update()
    pbar.close()

    return correct / N
