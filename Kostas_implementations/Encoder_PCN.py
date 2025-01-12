import torch
import torch.nn as nn


class MLP_to_encoder_output(nn.Module):

    def __init__(self, mlp_configs, drop=False, drop_prob=0.4):
        super().__init__()

        self.num_layers = len(mlp_configs) - 1
        self.DropoutProb = drop_prob

        layers = []
        for i in range(self.num_layers - 1):
            layers.append(nn.Linear(in_features=mlp_configs[i], out_features=mlp_configs[i + 1]))
            layers.append(nn.BatchNorm1d(mlp_configs[i + 1]))
            layers.append(nn.ReLU())

            if i != 0 and drop:
                layers.append(nn.Dropout(p=self.DropoutProb))

        ## Output layer
        layers.append(nn.Linear(in_features=mlp_configs[-2], out_features=mlp_configs[-1]))

        ### TODO: check that his is correct --> I probably need 1 more layer
        self.fnn_stack = nn.Sequential(*layers)

    def forward(self, x):
        fnn_output = self.fnn_stack(x)
        return fnn_output


class PCN_Encoder(nn.Module):
    def __init__(self, latent_dim_output, conv_configs, mlp_configs):
        super(PCN_Encoder, self).__init__()

        self.latent_dim_output = latent_dim_output

        ### First shared MLP block (PointNet layer)
        self.FirstConv = nn.Sequential(
            nn.Conv1d(conv_configs[0], conv_configs[1], 1),
            nn.BatchNorm1d(conv_configs[1]),
            nn.ReLU(inplace=True),
            nn.Conv1d(conv_configs[1], conv_configs[2], 1)
        )

        ### Second PointNet layer
        self.SecondConv = nn.Sequential(
            nn.Conv1d(conv_configs[3], conv_configs[3], 1),
            nn.BatchNorm1d(conv_configs[3]),
            nn.ReLU(inplace=True),
            nn.Conv1d(conv_configs[3], self.latent_dim_output, 1)
        )

        self.MLPToLatent = MLP_to_encoder_output(*mlp_configs)

    def forward(self, xyz):
        ### This should be the shape of the input: (batch, num_points, 3)
        B, N, _ = xyz.shape

        ### First computation of features
        feature = self.FirstConv(xyz.transpose(2, 1))  # (B,  50, N)
        ### Point-wise maxpool
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # (B,  50, 1)
        ### unpool --> skip connection
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)  # (B,  100, N)
        ### Second computation of features
        feature = self.SecondConv(feature)  # currently: (B, 50, N) --> Should be (B, >100, N)
        ### Point-wise maxpool
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # currently (B, 50) it should be (B, >100)

        y_encoded = self.MLPToLatent(feature_global)

        return y_encoded













