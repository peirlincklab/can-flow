import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


# class MetadataEncoder(nn.Module):
#     def __init__(self):
#         super(MetadataEncoder, self).__init__()
#
#         self.fc1 = nn.Linear(in_features=3, out_features=9).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
#
#     def forward(self, x):
#
#         x = F.gelu(self.fc1(x))
#
#         return x



class ConvEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(ConvEncoder, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=7, out_channels=64, kernel_size=(5, 4, 3), stride=2).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(4, 3, 2), stride=1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)

        self.flatten = nn.Flatten().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.softplus = nn.Softplus().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        self.fc_mean_logvar = nn.Linear(128 * 2 * 4 * 6 + 9, 2 * latent_dim).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, x, x_conf, metadata_enc, eps=1e-8):
        metadata_encoder = metadata_enc

        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))

        x = self.flatten(x)

        x_conf = metadata_encoder(x_conf)

        x = torch.cat((x, x_conf), dim=1)

        z_mean_logvar = self.fc_mean_logvar(x)

        mean, logvar = torch.chunk(z_mean_logvar, 2, dim=-1)

        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)

        return torch.distributions.MultivariateNormal(mean, scale_tril=scale_tril)


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(ConvDecoder, self).__init__()

        self.fc = nn.Linear(latent_dim + 9, 128 * 2 * 4 * 6).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        self.deconv1 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=(4, 3, 2), stride=1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.deconv2 = nn.ConvTranspose3d(in_channels=64, out_channels=4, kernel_size=(5, 4, 3), stride=2).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        self.bn1 = nn.BatchNorm3d(128)
        self.bn2 = nn.BatchNorm3d(64)

    def forward(self, z):
        x = F.gelu(self.fc(z))

        x = x.view(-1, 128, 2, 4, 6)

        x = F.gelu(self.deconv1(x))
        x = self.deconv2(x)

        return x


class ConvVAE(nn.Module):
    def __init__(self, latent_dim):
        super(ConvVAE, self).__init__()

        self.encoder = ConvEncoder(latent_dim).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.decoder = ConvDecoder(latent_dim).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        self.metadata_enc = MetadataEncoder().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def reparameterize(self, dist):

        return dist.rsample()

    def forward(self, x, x_conf):

        dist = self.encoder(x, x_conf, self.metadata_enc)

        z = self.reparameterize(dist)

        x_conf = self.metadata_enc(x_conf)

        z_concat = torch.cat((z, x_conf), dim=1)

        x_reconstruct = self.decoder(z_concat)

        return x_reconstruct, dist, self.metadata_enc


class TrainerConvVAE:
    def __init__(self, model, optimizer, epochs, train_loader, valid_loader):

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

    def loss_function(self, prediction, reference, dist, beta):

        r_loss_func = nn.MSELoss()
        loss_recon = r_loss_func(prediction.to(self.device), reference.to(self.device))
        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros(16, device=self.device), torch.eye(16, device=self.device)
        )


        loss_kl = torch.distributions.kl.kl_divergence(dist, std_normal).mean()

        loss = loss_recon + beta * loss_kl

        return loss, loss_recon, loss_kl

    def training(self):
        pbar = tqdm(total=self.epochs, desc="Epochs training...")
        beta = 2e-4
        best_val_loss = float('inf')
        best_model_weights = None
        for epoch in range(self.epochs):

            # if epoch >= 975:
            #     beta = 1e-3
            # elif epoch >= 700:
            #     beta = 0.05
            # else:
            #     beta = 1e-4

            ### Training phase

            self.model.train()
            train_total_loss = []
            train_recon_loss = []
            train_kl_loss = []
            for x_train, x_confounders in self.train_loader:

                x_train = x_train.to(self.device)
                x_confounders = x_confounders.to(self.device)

                self.optimizer.zero_grad()

                ### Forward propagation. Inputs to VAE are the batch of point clouds and the patient metadata
                prediction, dist, metadata_encoder = self.model(x_train, x_confounders)

                loss_train, loss_recon, kl_loss = self.loss_function(prediction=prediction,
                                                                     reference=x_train,
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
                for x_valid, x_confounders_valid in self.valid_loader:

                    x_valid = x_valid.to(self.device)
                    x_confounders_valid = x_confounders_valid.to(self.device)

                    prediction_valid, dist_valid, _ = self.model(x_valid, x_confounders_valid)

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

            ### Keep track of the model that results to the minimum validation error
            if ValidRecon_MeanBatch < best_val_loss:
                best_val_loss = ValidRecon_MeanBatch
                best_model_weights = self.model.state_dict()

            print(f"\nEpoch   Training   Validation    Validation MSE    Beta\n"
                  f"{epoch}   {TrainTotal_MeanBatch}  {ValidTotal_MeanBatch}  {ValidRecon_MeanBatch}  {beta}\n"
                  f"====================================================")


            pbar.update()
        pbar.close()
        print("Done training!")

        if best_model_weights:
            self.model.load_state_dict(best_model_weights)

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

        return self.model, metadata_encoder

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

