import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class VAE_Encoder(nn.Module):
    def __init__(self, latent_dim, cond_dim):
        super().__init__()

        self.mlp_conf = nn.Linear(in_features=cond_dim, out_features=720 * 1)

        self.conv1 = nn.Conv3d(in_channels=4, out_channels=32, kernel_size=(1, 1, 1), stride=1)  ### 3 + 1 for condition
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(2, 2, 2), stride=1)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(1, 1, 1), stride=1)
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(2, 2, 2), stride=1)

        self.flatten = nn.Flatten()

        self.fc_latent = nn.Linear(128 * 6 * 7 * 8, 2 * latent_dim)

    def forward(self, x, x_conf):
        conf_embedding = F.gelu(self.mlp_conf(x_conf))
        conf_embedding = conf_embedding.reshape(-1, 1, 8, 9, 10)

        x = torch.cat((x, conf_embedding), dim=1)

        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = F.gelu(self.conv4(x))

        x = self.flatten(x)

        z_latent = self.fc_latent(x)

        mu, logvar = torch.chunk(z_latent, 2, dim=1)

        return mu, logvar


class VAE_Decoder(nn.Module):
    def __init__(self, latent_dim, in_dim_conf, embed_dim_conf):
        super().__init__()

        self.conf_embedding = nn.Sequential(
            nn.Linear(in_dim_conf, embed_dim_conf),
            nn.GELU(),
            nn.Linear(embed_dim_conf, embed_dim_conf)
        )

        self.fc = nn.Linear(latent_dim + embed_dim_conf, 128 * 6 * 7 * 8)

        self.deconv1 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=(2, 2, 2), stride=1)
        self.deconv2 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=(1, 1, 1), stride=1)
        self.deconv3 = nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=(2, 2, 2), stride=1)
        self.deconv4 = nn.ConvTranspose3d(in_channels=32, out_channels=3, kernel_size=(1, 1, 1), stride=1)

    def forward(self, z, x_conf):
        x_conf_embed = self.conf_embedding(x_conf)

        x = torch.cat((z, x_conf_embed), dim=1)

        x = F.gelu(self.fc(x))

        x = x.view(-1, 128, 6, 7, 8)

        x = F.gelu(self.deconv1(x))
        x = F.gelu(self.deconv2(x))
        x = F.gelu(self.deconv3(x))

        x = self.deconv4(x)

        return x


class cVAE(nn.Module):
    def __init__(self, latent_dim, cond_dim, embed_dim_conf):
        super().__init__()

        self.encoder = VAE_Encoder(latent_dim, cond_dim)
        self.decoder = VAE_Decoder(latent_dim, cond_dim, embed_dim_conf)

    def reparametrize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mean + eps * std

    def forward(self, x, x_conf):
        mu, logvar = self.encoder(x, x_conf)

        z = self.reparametrize(mu, logvar)

        x_recon = self.decoder(z, x_conf)

        return x_recon, mu, logvar


class Trainer_cVAE:
    def __init__(self, model, optimizer, epochs, train_loader, valid_loader, latent_dim, beta):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        self.latent_dimension = latent_dim
        self.beta = beta

    def loss_function(self, prediction, reference, mean, logvar):

        r_loss_func = nn.MSELoss()

        loss_recon = r_loss_func(prediction.to(self.device), reference.to(self.device))

        loss_kl = -0.5 * torch.mean(torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1))

        loss = loss_recon + self.beta * loss_kl

        return loss, loss_recon, loss_kl

    def training(self):
        pbar = tqdm(total=self.epochs, desc="Epochs training...")
        best_val_loss = float('inf')
        best_model_weights = None
        for epoch in range(self.epochs):
            ### Training phase

            self.model.to(self.device)
            self.model.train()
            train_total_loss = []
            train_recon_loss = []
            train_kl_loss = []
            for x_train, x_conf in self.train_loader:
                x_train = x_train.to(self.device)
                x_conf = x_conf.to(self.device)

                self.optimizer.zero_grad()

                ### Forward propagation
                prediction, mu, logvar = self.model(x_train, x_conf)

                loss_train, loss_recon, loss_kl = self.loss_function(prediction=prediction, reference=x_train,
                                                                     mean=mu, logvar=logvar)

                loss_train.backward()
                self.optimizer.step()

                train_total_loss.append(loss_train.item())
                train_recon_loss.append(loss_recon.item())
                train_kl_loss.append(loss_kl.item())

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
                for x_valid, x_conf_valid in self.valid_loader:
                    x_valid = x_valid.to(self.device)
                    x_conf_valid = x_conf_valid.to(self.device)

                    prediction_valid, mu_valid, logvar_valid = self.model(x_valid, x_conf_valid)

                    loss_valid, loss_recon_valid, loss_kl_valid = self.loss_function(prediction=prediction_valid,
                                                                                     reference=x_valid,
                                                                                     mean=mu_valid,
                                                                                     logvar=logvar_valid)

                    valid_total_loss.append(loss_valid.item())
                    valid_recon_loss.append(loss_recon_valid.item())
                    valid_kl_loss.append(loss_kl_valid.item())

                ValidTotal_MeanBatch = np.mean(valid_total_loss)
                ValidRecon_MeanBatch = np.mean(valid_recon_loss)
                ValidKL_MeanBatch = np.mean(valid_kl_loss)

                self.ValidTotal_MeanBatch_Epochs.append(ValidTotal_MeanBatch)
                self.ValidRecon_MeanBatch_Epochs.append(ValidRecon_MeanBatch)
                self.ValidKL_MeanBatch_Epochs.append(ValidKL_MeanBatch)

            ### Keep track of the model that results to the minimum validation error
            if ValidTotal_MeanBatch < best_val_loss:
                best_val_loss = ValidTotal_MeanBatch
                best_model_weights = self.model.state_dict()

            print(f"\nEpoch     Training     Validation     Validation MSE     Beta\n"
                  f"{epoch}     {TrainTotal_MeanBatch}     {ValidTotal_MeanBatch}     {ValidRecon_MeanBatch}     {self.beta} \n"
                  f"====================================================")

            pbar.update()
        pbar.close()
        print("Done training!")

        if best_model_weights:
            self.model.load_state_dict(best_model_weights)

        return self.model

    def plot_losses(self):

        ### Plot the losses
        plt.figure()
        plt.loglog(self.TrainTotal_MeanBatch_Epochs, label='Total error')
        plt.loglog(self.TrainRecon_MeanBatch_Epochs, label='Reconstruction')
        plt.loglog(self.TrainKL_MeanBatch_Epochs, label='KL')
        plt.title("Training")
        plt.xlabel("# Epochs")
        plt.ylabel("Loss function")
        plt.legend()
        plt.show()

        plt.figure()
        plt.loglog(self.ValidTotal_MeanBatch_Epochs, label='Total error')
        plt.loglog(self.ValidRecon_MeanBatch_Epochs, label='Reconstruction')
        plt.loglog(self.ValidKL_MeanBatch_Epochs, label='KL')
        plt.xlabel("# Epochs")
        plt.ylabel("Loss function")
        plt.legend()
        plt.show()
