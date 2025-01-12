import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from PointNet_helpers import PointNetEncoder, feature_transform_reguliarzer


class PointNet_Encoder(nn.Module):
    def __init__(self, k=40):
        super(PointNet_Encoder, self).__init__()
        channel = 3 ### we operate in the 3d space
        ### Use the PointNet, before the MLP
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        ### "k" refers to the latent representation in our case
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        ### trans --> coordinate transformation of points / trans_feat --> transformation (alignment) of features
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x))) ### elegant way to write the NN layers --> don't need a whole new class
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))

        y_encoded = self.fc3(x)
        ### Currently, "y_encoded" is the representation that is outputed from the encoder.
        ### This will be concatenated with the confounder information
        return y_encoded, trans_feat


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss