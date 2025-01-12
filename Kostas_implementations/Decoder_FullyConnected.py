import torch
import torch.nn as nn


class FC_Decoder(nn.Module):

    def __init__(self, num_points, latent_dim, fc_decoder, batchnorm=True):
        super().__init__()

        self.num_points = num_points
        self.fc_decoder_configs = fc_decoder

        self.fc1 = nn.Linear(latent_dim, self.fc_decoder_configs[0])
        if batchnorm:
            self.bn1 = nn.BatchNorm1d(self.fc_decoder_configs[0])

        self.fc2 = nn.Linear(self.fc_decoder_configs[0], self.fc_decoder_configs[1])
        if batchnorm:
            self.bn2 = nn.BatchNorm1d(self.fc_decoder_configs[1])

        self.fc3 = nn.Linear(self.fc_decoder_configs[1], num_points * 3)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))

        ## Not nonlinearity in the last layer
        x = self.fc3(x)

        x = x.view(batch_size, -1, 3)

        return x
