import torch
import torch.nn as nn


class PCN_Decoder(nn.Module):
    """
    Implementation of the Point Completion Network
    Attributes:
        num_dense:  2730
        encoder output dim: 170
        grid_size:  4
        num_coarse: 170
    """

    def __init__(self, num_dense, grid_size, configs_mlp):
        super().__init__()

        self.num_dense = num_dense ### num of points in the full point cloud
        self.grid_size = grid_size ### grid size for the decoder
        # self.latent_vector = latent_representation ### will be the input to the decoder
        self.latent_dim = 16  ### latent dimension --> set to 15 for now but will make it dynamic

        assert self.num_dense % self.grid_size ** 2 == 0

        self.num_coarse = self.num_dense // (self.grid_size ** 2)

        ### mlp to be used in the decoder, to reconstruct the coarse point cloud
        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, configs_mlp[0]),
            nn.ReLU(inplace=True),
            nn.Linear(configs_mlp[0], configs_mlp[0]),
            nn.ReLU(inplace=True),
            nn.Linear(configs_mlp[0], 3 * self.num_coarse)
        )

        self.final_conv = nn.Sequential(
            nn.Conv1d(self.latent_dim + 2 + 3, configs_mlp[1], 1),
            nn.BatchNorm1d(configs_mlp[1]),
            nn.ReLU(inplace=True),
            nn.Conv1d(configs_mlp[1], configs_mlp[1], 1),
            nn.BatchNorm1d(configs_mlp[1]),
            nn.ReLU(inplace=True),
            nn.Conv1d(configs_mlp[1], 3, 1)
        )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(
            self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(
            self.grid_size, self.grid_size).reshape(1, -1)

        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2)  # (1, 2, S)

    def forward(self, xyz, latent_vector):
        ### (batch_size, num_points, 3) <-- this will be the input to the PCN decoder
        B, N, _ = xyz.shape

        coarse = self.mlp(latent_vector).reshape(-1, self.num_coarse, 3)  # (B, num_coarse, 3), coarse point cloud
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)  # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)  # (B, 3, num_fine)

        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)  # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)  # (B, 2, num_fine)

        feature_global = latent_vector.unsqueeze(2).expand(-1, -1, self.num_dense)  # currently this is (B, 16, num_fine). Should it be something like (B, 1024, num_fine)?
        feat = torch.cat([feature_global, seed, point_feat], dim=1)  # (B, 1024+2+3, num_fine)

        fine = self.final_conv(feat) + point_feat  # (B, 3, num_fine), fine point cloud

        ### the output for the fine point cloud will be (batch, num_fine, 3)
        return coarse.contiguous(), fine.transpose(1, 2).contiguous()