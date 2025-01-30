import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class EnsembleVAE(nn.Module):
    def __init__(self, vae_models):
        super(EnsembleVAE, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.vae_models_trained = vae_models
        self.raw_weights = nn.Parameter(torch.ones(len(vae_models), device=self.device), requires_grad=True).to(self.device)

    def forward(self, x):
        weights = F.softmax(self.raw_weights)

        self.weight_matrices = [torch.diag(torch.ones(32, device=self.device)*weight) for weight in weights]

        reconstructions_weighted = []
        dist_means_weighted = []
        dist_vars_weighted = []
        for i, model in enumerate(self.vae_models_trained):
            model.to(self.device)
            model.eval()
            with torch.no_grad():
                x = x.to(self.device)
                reconstruction, distribution = model(x)

            reconstructions_weighted.append(reconstruction * self.weight_matrices[i][0, 0])

            dist_means_weighted.append(distribution.mean @ self.weight_matrices[i])
            dist_vars_weighted.append(self.weight_matrices[i] @ distribution.covariance_matrix @ self.weight_matrices[i].T)

        print(weights)
        # Combine reconstructions and latents using the ensemble weights
        ensemble_recon = sum(reconstructions_weighted)

        ensemble_mean = sum(dist_means_weighted)
        ensemble_variance = sum(dist_vars_weighted)

        ensemble_dist = torch.distributions.MultivariateNormal(ensemble_mean, ensemble_variance)

        return ensemble_recon, ensemble_dist

    def generation(self, z):

        shapes = []
        for i, model in enumerate(self.vae_models_trained):
            model.to(self.device)
            model.eval()
            with torch.no_grad():
                z = z.to(self.device)
                shape = model.decoder(z)
                shapes.append(shape * self.weight_matrices[i][0, 0])

        ensemble_shape = sum(shapes)

        return ensemble_shape


class SiameseNetEnc(nn.Module):
    def __init__(self):
        super(SiameseNetEnc, self).__init__()

        self.fc1 = nn.Linear(in_features=2730, out_features=1024).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.fc2 = nn.Linear(in_features=1024, out_features=512).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))

        return x


class SiameseNetDec(nn.Module):
    def __init__(self):
        super(SiameseNetDec, self).__init__()

        self.fc1 = nn.Linear(in_features=512, out_features=1024).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.fc2 = nn.Linear(in_features=1024, out_features=2730).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)

        return x


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.x_encoder = SiameseNetEnc().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.y_encoder = SiameseNetEnc().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.z_encoder = SiameseNetEnc().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        self.latent_dim = latent_dim
        self.softplus = nn.Softplus().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # self.fc_mean_logvar = nn.Linear(in_features=512 * 3, out_features=2 * latent_dim)
        self.fc_mean_logvar = nn.Linear(in_features=512, out_features=2 * latent_dim).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, obj, eps=1e-8):
        x = obj[:, :, 0]
        y = obj[:, :, 1]
        z = obj[:, :, 2]

        x = self.x_encoder(x)
        y = self.y_encoder(y)
        z = self.z_encoder(z)

        # intermediate_output = torch.cat((x, y, z), dim=1)

        intermediate_output = x + y + z

        z_mean_logvar = self.fc_mean_logvar(intermediate_output)

        mean, logvar = torch.chunk(z_mean_logvar, 2, dim=-1)

        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)

        return torch.distributions.MultivariateNormal(mean, scale_tril=scale_tril)


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        self.fc_intermed = nn.Linear(in_features=latent_dim, out_features=512 * 3).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        self.x_decoder = SiameseNetDec().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.y_decoder = SiameseNetDec().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.z_decoder = SiameseNetDec().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, latent):
        x_intermed = F.gelu(self.fc_intermed(latent))

        x, y, z = torch.chunk(x_intermed, 3, dim=-1)

        x = self.x_decoder(x)
        y = self.y_decoder(y)
        z = self.z_decoder(z)

        ### TODO: Check this one
        shape = torch.cat((x, y, z), dim=1).reshape(-1, 2730, 3)

        return shape


class SiameseVAE(nn.Module):
    def __init__(self, latent_dim):
        super(SiameseVAE, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = Encoder(latent_dim=latent_dim).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.decoder = Decoder(latent_dim=latent_dim).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def reparameterize(self, dist):
        return dist.rsample()

    def forward(self, x):
        dist = self.encoder(x)

        z = self.reparameterize(dist)

        x_reconstruct = self.decoder(z)

        return x_reconstruct, dist


class TrainerSiamese:
    def __init__(self, model, optimizer, epochs, train_loader, valid_loader):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.TrainTotal_MeanBatch_Epochs = []
        self.TrainRecon_MeanBatch_Epochs = []
        self.TrainKL_MeanBatch_Epochs = []

        self.ValidTotal_MeanBatch_Epochs = []
        self.ValidRecon_MeanBatch_Epochs = []
        self.ValidKL_MeanBatch_Epochs = []

    def beta_schedule(self, epoch, anneal=False, beta=None):

        if anneal:
            if epoch >= 450:
                beta = 1e-4
            elif epoch >= 300:
                beta = 0.25
            else:
                beta = 1e-4
        else:
            beta = beta

        return beta

    def loss_function(self, prediction, reference, dist, beta):

        r_loss_func = nn.MSELoss()
        loss_recon = r_loss_func(prediction.to(self.device), reference.to(self.device))

        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros(32, device=self.device), torch.eye(32, device=self.device)
        )

        loss_kl = torch.distributions.kl.kl_divergence(dist, std_normal).mean()

        loss = loss_recon + beta * loss_kl

        return loss, loss_recon, loss_kl

    def training(self, anneal=None, beta=None):

        pbar = tqdm(total=self.epochs, desc="Epochs training...")
        for epoch in range(self.epochs):

            beta = self.beta_schedule(epoch=epoch, anneal=anneal, beta=beta)

            # if epoch >= 965:
            #     beta = 1e-6
            # elif epoch >= 700:
            #     beta = 0.2
            # else:
            #     beta = 1e-4

            self.model.train()
            train_total_loss = []
            train_recon_loss = []
            train_kl_loss = []
            for x_train, _ in self.train_loader:
                x_train = x_train.to(self.device)

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
                    x_valid = x_valid.to(self.device)

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


class TrainerEnsembleVAE:
    def __init__(self, model, optimizer, epochs, train_loader, valid_loader):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ### Now this model refers to the ensemble model
        self.model = model

        ### Now the optimizer refers to the weights of the ensemble VAE.
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

    def beta_schedule(self, epoch, anneal=False, beta=None):

        if anneal:
            if epoch >= 660:
                beta = 1e-4
            elif epoch >= 300:
                beta = 0.25
            else:
                beta = 1e-4
        else:
            beta = beta

        return beta

    def loss_function(self, prediction, reference, dist, beta=1):
        r_loss_func = nn.MSELoss()
        loss_recon = r_loss_func(prediction.to(self.device), reference.to(self.device))

        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros(32, device=self.device), torch.eye(32, device=self.device)
        )

        loss_kl = torch.distributions.kl.kl_divergence(dist, std_normal).mean()

        loss = loss_recon + beta * loss_kl

        return loss, loss_recon, loss_kl

    def training(self, anneal, beta):

        pbar = tqdm(total=self.epochs, desc="Epochs training...")
        for epoch in range(self.epochs):

            beta = self.beta_schedule(epoch=epoch, anneal=anneal, beta=beta)

            train_total_loss = []
            train_recon_loss = []
            train_kl_loss = []
            for x_train, _ in self.train_loader:
                x_train = x_train.to(self.device)
                self.optimizer.zero_grad()

                ensemble_recon, ensemble_dist = self.model(x_train)

                loss_train, loss_recon, loss_kl = self.loss_function(prediction=ensemble_recon, reference=x_train,
                                                                     dist=ensemble_dist, beta=beta)

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
            with torch.no_grad():

                valid_total_loss = []
                valid_recon_loss = []
                valid_kl_loss = []
                for x_valid, _ in self.valid_loader:
                    x_valid = x_valid.to(self.device)

                    ensemble_recon_valid, ensemble_dist_valid = self.model(x_valid)

                    loss_valid, loss_recon_valid, loss_kl_valid = self.loss_function(prediction=ensemble_recon_valid,
                                                                                     reference=x_valid,
                                                                                     dist=ensemble_dist_valid,
                                                                                     beta=beta)

                    valid_total_loss.append(loss_valid.item())
                    valid_recon_loss.append(loss_recon_valid.item())
                    valid_kl_loss.append(loss_kl_valid.item())

                ValidTotal_MeanBatch = np.mean(valid_total_loss)
                ValidRecon_MeanBatch = np.mean(valid_recon_loss)
                ValidKL_MeanBatch = np.mean(valid_kl_loss)

                self.ValidTotal_MeanBatch_Epochs.append(ValidTotal_MeanBatch)
                self.ValidRecon_MeanBatch_Epochs.append(ValidRecon_MeanBatch)
                self.ValidKL_MeanBatch_Epochs.append(ValidKL_MeanBatch)

            print(f"\nEpoch   Training   Validation    Validation MSE\n"
                  f"{epoch}   {TrainTotal_MeanBatch}  {ValidTotal_MeanBatch}  {ValidRecon_MeanBatch}\n"
                  f"====================================================")

            pbar.update()
        pbar.close()
        print("Done training!")

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