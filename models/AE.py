import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=(1, 1, 1), stride=1).to('cuda')
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(2, 2, 2), stride=1).to('cuda')
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(1, 1, 1), stride=1).to('cuda')
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(2, 2, 2), stride=1).to('cuda')

        self.flatten = nn.Flatten().to('cuda')

        self.fc_latent = nn.Linear(128 * 6 * 7 * 8, latent_dim).to('cuda')


    def forward(self, x):

        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = F.gelu(self.conv4(x))

        x = self.flatten(x)

        z_latent = self.fc_latent(x)

        return z_latent


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 128 * 6 * 7 * 8).to('cuda')

        self.deconv1 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=(2, 2, 2), stride=1).to('cuda')
        self.deconv2 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=(1, 1, 1), stride=1).to('cuda')
        self.deconv3 = nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=(2, 2, 2), stride=1).to('cuda')
        self.deconv4 = nn.ConvTranspose3d(in_channels=32, out_channels=3, kernel_size=(1, 1, 1), stride=1).to('cuda')


    def forward(self, z):

        x = F.gelu(self.fc(z))

        x = x.view(-1, 128, 6, 7, 8)

        x = F.gelu(self.deconv1(x))
        x = F.gelu(self.deconv2(x))
        x = F.gelu(self.deconv3(x))

        x = self.deconv4(x)

        return x


class ConvAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):

        z_latent = self.encoder(x)
        out_shape = self.decoder(z_latent)

        return out_shape


class TrainerAE:
    def __init__(self, model, optimizer, epochs, train_loader, valid_loader, latent_dim):

        self.device = torch.device('cuda')

        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.TrainTotal_MeanBatch_Epochs = []
        self.ValidTotal_MeanBatch_Epochs = []

        self.latent_dimension = latent_dim


    def loss_function(self, prediction, reference):

        r_loss_func = nn.MSELoss()

        loss_recon = r_loss_func(prediction.to(self.device), reference.to(self.device))

        return loss_recon

    def training(self):
        pbar = tqdm(total=self.epochs, desc="Epochs training...")
        best_val_loss = float('inf')
        best_model_weights = None
        for epoch in range(self.epochs):
            ### Training phase

            self.model.train()
            train_total_loss = []
            for x_train in self.train_loader:

                x_train = x_train.to(self.device)

                self.optimizer.zero_grad()

                ### Forward propagation
                prediction = self.model(x_train)

                loss_train = self.loss_function(prediction=prediction, reference=x_train)

                loss_train.backward()
                self.optimizer.step()

                train_total_loss.append(loss_train.item())

            TrainTotal_MeanBatch = np.mean(train_total_loss)

            self.TrainTotal_MeanBatch_Epochs.append(TrainTotal_MeanBatch)

            ### Validation phase
            self.model.eval()
            valid_total_loss = []
            with torch.no_grad():
                for x_valid in self.valid_loader:

                    x_valid = x_valid.to(self.device)

                    prediction_valid = self.model(x_valid)

                    loss_valid = self.loss_function(prediction=prediction_valid, reference=x_valid)

                    valid_total_loss.append(loss_valid.item())

                ValidTotal_MeanBatch = np.mean(valid_total_loss)

                self.ValidTotal_MeanBatch_Epochs.append(ValidTotal_MeanBatch)
            ### Keep track of the model that results to the minimum validation error
            if ValidTotal_MeanBatch < best_val_loss:
                best_val_loss = ValidTotal_MeanBatch
                best_model_weights = self.model.state_dict()


            print(f"\nEpoch   Training   Validation\n"
                  f"{epoch}   {TrainTotal_MeanBatch}  {ValidTotal_MeanBatch} \n"
                  f"====================================================")


            pbar.update()
        pbar.close()
        print("Done training!")

        if best_model_weights:
            self.model.load_state_dict(best_model_weights)

        return self.model


    def plot_losses(self):

        plt.figure()
        plt.loglog(self.TrainTotal_MeanBatch_Epochs, label='Training')
        plt.loglog(self.ValidTotal_MeanBatch_Epochs, label='Validation')
        plt.title("Training")
        plt.xlabel("# Epochs")
        plt.ylabel("Loss function")
        plt.legend()
        plt.show()
