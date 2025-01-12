import torch.nn as nn
import torch.nn.functional as F
from PointNet2_helpers import PointNetSetAbstraction


class PointNet2_Encoder(nn.Module):
    def __init__(self, num_latent, additional_channel=True, num_additional=3):
        super(PointNet2_Encoder, self).__init__()
        in_channel = num_additional + 3 if additional_channel else 3
        self.normal_channel = additional_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=10, nsample=8, in_channel=in_channel,
                                          mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=20, nsample=16, in_channel=128 + 3,
                                          mlp=[64, 64, 128], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=128 + 3,
                                          mlp=[64, 128, 256], group_all=True)
        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(64, num_latent)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 256)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return x
