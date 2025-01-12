import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(ConvEncoder, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(5, 4, 3), stride=2)
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(4, 3, 2), stride=1)

        # self.bn1 = nn.BatchNorm3d(32)
        # self.bn2 = nn.BatchNorm3d(52)

        self.flatten = nn.Flatten()
        self.softplus = nn.Softplus()

        self.fc_mean_logvar = nn.Linear(128 * 2 * 4 * 6, 2 * latent_dim)

    def forward(self, x, eps=1e-8):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))

        x = self.flatten(x)

        z_mean_logvar = self.fc_mean_logvar(x)

        mean, logvar = torch.chunk(z_mean_logvar, 2, dim=-1)

        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)

        return torch.distributions.MultivariateNormal(mean, scale_tril=scale_tril)


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(ConvDecoder, self).__init__()

        self.fc = nn.Linear(latent_dim, 128 * 2 * 4 * 6)

        self.deconv1 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=(4, 3, 2), stride=1)
        self.deconv2 = nn.ConvTranspose3d(in_channels=64, out_channels=3, kernel_size=(5, 4, 3), stride=2)

    # self.bn1 = nn.BatchNorm3d(32)

    def forward(self, z):
        x = F.gelu(self.fc(z))

        x = x.view(-1, 128, 2, 4, 6)

        x = F.gelu(self.deconv1(x))
        x = self.deconv2(x)

        return x


class ConvVAE(nn.Module):
    def __init__(self, latent_dim):
        super(ConvVAE, self).__init__()

        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)

    def reparameterize(self, dist):

        return dist.rsample()

    def forward(self, x):
        dist = self.encoder(x)

        z = self.reparameterize(dist)

        x_reconstruct = self.decoder(z)

        return x_reconstruct, dist


class TrainerConvVAE:
    def __init__(self, model, optimizer, epochs, train_loader, valid_loader):

        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.TrainTotal_MeanBatch_Epochs = []
        self.TrainRecon_MeanBatch_Epochs = []
        self.TrainKL_MeanBatch_Epochs = []

        self.ValidTotal_MeanBatch_Epochs = []
        self.ValidRecon_MeanBatch_Epochs = []
        self.ValidKL_MeanBatch_Epochs = []

    def loss_function(self, prediction, reference, dist, beta):

        r_loss_func = nn.MSELoss()
        loss_recon = r_loss_func(prediction, reference)

        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros(32), torch.eye(32)
        )

        loss_kl = torch.distributions.kl.kl_divergence(dist, std_normal).mean()

        loss = loss_recon + beta * loss_kl

        return loss, loss_recon, loss_kl

    def training(self):
        pbar = tqdm(total=self.epochs, desc="Epochs training...")
        # beta = 0.05
        for epoch in range(self.epochs):

            if epoch >= 975:
                beta = 1e-3
            elif epoch >= 700:
                beta = 0.05
            else:
                beta = 1e-4

            ### Training phase
            ### Annealing of beta term, used in the loss of the beta-VAE
            # if epoch >= 1950:
            #     beta = 1e-3
            # elif epoch >= 1500:
            #     beta = 0.25
            # elif epoch >= 1300:
            #     beta = 1e-4
            # elif epoch >= 1000:
            #     beta = 0.5
            # else:
            #     beta = 1e-5

            self.model.train()
            train_total_loss = []
            train_recon_loss = []
            train_kl_loss = []
            for x_train, _ in self.train_loader:
                self.optimizer.zero_grad()

                ### Forward propagation. Inputs to VAE are the batch of point clouds and the patient metadata
                prediction, dist = self.model(x_train)

                loss_train, loss_recon, kl_loss = self.loss_function(prediction=prediction, reference=x_train,
                                                                     dist=dist, beta=beta)

                loss_train.backward()
                self.optimizer.step()

                train_total_loss.append(loss_train.item())
                train_recon_loss.append(loss_recon.item())
                train_kl_loss.append(kl_loss.item())

            TrainTotal_MeanBatch = np.mean(train_total_loss)
            TrainRecon_MeanBatch = np.mean(train_recon_loss)
            TrainKL_MeanBatch = np.mean(train_kl_loss)

            self.TrainTotal_MeanBatch_Epochs.append(TrainTotal_MeanBatch)
            self.TrainRecon_MeanBatch_Epochs.append(TrainRecon_MeanBatch)
            self.TrainKL_MeanBatch_Epochs.append(TrainKL_MeanBatch)

            ### Validation phase
            self.model.eval()
            valid_total_loss = []
            valid_recon_loss = []
            valid_kl_loss = []

            with torch.no_grad():
                for x_valid, _ in self.valid_loader:
                    prediction_valid, dist_valid = self.model(x_valid)

                    loss_valid, loss_recon_valid, loss_kl_valid = self.loss_function(prediction=prediction_valid,
                                                                                     reference=x_valid,
                                                                                     dist=dist_valid, beta=beta)

                    valid_total_loss.append(loss_valid.item())
                    valid_recon_loss.append(loss_recon_valid.item())
                    valid_kl_loss.append(loss_kl_valid.item())

                ValidTotal_MeanBatch = np.mean(valid_total_loss)
                ValidRecon_MeanBatch = np.mean(valid_recon_loss)
                ValidKL_MeanBatch = np.mean(valid_kl_loss)

                self.ValidTotal_MeanBatch_Epochs.append(ValidTotal_MeanBatch)
                self.ValidRecon_MeanBatch_Epochs.append(ValidRecon_MeanBatch)
                self.ValidKL_MeanBatch_Epochs.append(ValidKL_MeanBatch)

            print(f"\nEpoch   Training   Validation    Validation MSE    Beta\n"
                  f"{epoch}   {TrainTotal_MeanBatch}  {ValidTotal_MeanBatch}  {ValidRecon_MeanBatch}  {beta}\n"
                  f"====================================================")

            pbar.update()
        pbar.close()
        print("Done training!")

        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.view(-1).cpu().numpy())

        # Flatten and plot
        gradients = np.concatenate(gradients)
        plt.hist(gradients, bins=50)
        plt.title("Gradient Magnitude Distribution")
        plt.xlabel("Gradient Value")
        plt.ylabel("Frequency")
        plt.show()

        return self.model

    def plot_losses(self):

        ### Plot the losses
        plt.figure()
        plt.semilogy(self.TrainTotal_MeanBatch_Epochs, label='Total error')
        plt.semilogy(self.TrainRecon_MeanBatch_Epochs, label='Reconstruction')
        plt.semilogy(self.TrainKL_MeanBatch_Epochs, label='KL')
        plt.title("Training")
        plt.xlabel("# Epochs")
        plt.ylabel("Loss function")
        plt.legend()
        plt.show()

        plt.figure()
        plt.semilogy(self.ValidTotal_MeanBatch_Epochs, label='Total error')
        plt.semilogy(self.ValidRecon_MeanBatch_Epochs, label='Reconstruction')
        plt.semilogy(self.ValidKL_MeanBatch_Epochs, label='KL')
        plt.xlabel("# Epochs")
        plt.ylabel("Loss function")
        plt.legend()
        plt.show()

