import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class EnsembleVAE(nn.Module):
    def __init__(self, vae_models):
        super(EnsembleVAE, self).__init__()

        self.vae_models_trained = vae_models

    def forward(self, x):

        reconstructions = []
        distributions = []
        for model in self.vae_models_trained:
            model.eval()
            with torch.no_grad():

                reconstruction, distribution = model(x)

                reconstructions.append(reconstruction)
                distributions.append(distribution)

        return reconstructions, distributions


class SiameseNetEnc(nn.Module):
    def __init__(self):
        super(SiameseNetEnc, self).__init__()

        self.fc1 = nn.Linear(in_features=2730, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)

    def forward(self, x):

        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))

        return x


class SiameseNetDec(nn.Module):
    def __init__(self):
        super(SiameseNetDec, self).__init__()

        self.fc1 = nn.Linear(in_features=512, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=2730)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)

        return x


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.x_encoder = SiameseNetEnc()
        self.y_encoder = SiameseNetEnc()
        self.z_encoder = SiameseNetEnc()

        self.latent_dim = latent_dim
        self.softplus = nn.Softplus()

        self.fc_mean_logvar = nn.Linear(in_features=512 * 3, out_features=2 * latent_dim)

    def forward(self, obj, eps=1e-8):

        x = obj[:, :, 0]
        y = obj[:, :, 1]
        z = obj[:, :, 2]

        x = self.x_encoder(x)
        y = self.y_encoder(y)
        z = self.z_encoder(z)

        intermediate_output = torch.cat((x, y, z), dim=1)

        z_mean_logvar = self.fc_mean_logvar(intermediate_output)

        mean, logvar = torch.chunk(z_mean_logvar, 2, dim=-1)

        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)

        return torch.distributions.MultivariateNormal(mean, scale_tril=scale_tril)


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        self.fc_intermed = nn.Linear(in_features=latent_dim, out_features=512 * 3)

        self.x_decoder = SiameseNetDec()
        self.y_decoder = SiameseNetDec()
        self.z_decoder = SiameseNetDec()

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

        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)

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

        self.TrainTotal_MeanBatch_Epochs = []
        self.TrainRecon_MeanBatch_Epochs = []
        self.TrainKL_MeanBatch_Epochs = []

        self.ValidTotal_MeanBatch_Epochs = []
        self.ValidRecon_MeanBatch_Epochs = []
        self.ValidKL_MeanBatch_Epochs = []

    def beta_schedule(self, epoch, anneal=False, beta=None):

        if anneal:
            if epoch >= 665:
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
        loss_recon = r_loss_func(prediction, reference)

        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros(32), torch.eye(32)
        )

        loss_kl = torch.distributions.kl.kl_divergence(dist, std_normal).mean()

        loss = loss_recon + beta * loss_kl

        return loss, loss_recon, loss_kl

    def training(self, anneal, beta):
        best_val_loss = float('inf')
        best_model_weights = None

        pbar = tqdm(total=self.epochs, desc="Epochs training...")
        for epoch in range(self.epochs):

            beta = self.beta_schedule(epoch=epoch, anneal=anneal, beta=beta)

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

            if ValidTotal_MeanBatch - ValidRecon_MeanBatch < best_val_loss:
                best_val_loss = ValidTotal_MeanBatch - ValidRecon_MeanBatch
                best_model_weights = self.model.state_dict()

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
    def __init__(self, model, weights, optimizer, epochs, train_loader, valid_loader):

        ### Now this model refers to the ensemble model
        self.model = model
        self.weights = weights

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

    def loss_function(self, prediction, reference, dist, beta=1):
        r_loss_func = nn.MSELoss()
        loss_recon = r_loss_func(prediction, reference)

        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros(32), torch.eye(32)
        )

        loss_kl = torch.distributions.kl.kl_divergence(dist, std_normal).mean()

        loss = loss_recon + beta * loss_kl

        return loss, loss_recon, loss_kl

    def training(self):

        best_val_loss = float('inf')
        best_model_weights = None

        pbar = tqdm(total=self.epochs, desc="Epochs training...")
        for epoch in range(self.epochs):

            train_total_loss = []
            train_recon_loss = []
            train_kl_loss = []
            for x_train, _ in self.train_loader:
                self.optimizer.zero_grad()
                with torch.no_grad():
                    reconstructions, distributions = self.model(x_train)

                losses = [self.loss_function(prediction=reconstructions[i], reference=x_train, dist=distributions[i])
                          for i in range(len(reconstructions))]

                losses_total = [losses[i][0] for i in range(len(losses))]
                losses_recon = [losses[i][1] for i in range(len(losses))]
                losses_kl = [losses[i][2] for i in range(len(losses))]

                self.weights = F.softmax(self.weights, dim=0)

                ensemble_loss = torch.dot(torch.tensor(losses_total, dtype=torch.float32), self.weights)

                ensemble_loss_recon = torch.dot(torch.tensor(losses_recon, dtype=torch.float32), self.weights)
                ensemble_loss_kl = torch.dot(torch.tensor(losses_kl, dtype=torch.float32), self.weights)

                ensemble_loss.backward(retain_graph=True)
                self.optimizer.step()

                train_total_loss.append(ensemble_loss.item())
                train_recon_loss.append(ensemble_loss_recon.item())
                train_kl_loss.append(ensemble_loss_kl.item())

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
                    reconstructions_valid, distributions_valid = self.model(x_valid)

                    losses_valid = [
                        self.loss_function(
                            prediction=reconstructions_valid[i], reference=x_valid, dist=distributions_valid[i]
                        )
                        for i in range(len(reconstructions_valid))
                    ]

                    losses_total_valid = [losses_valid[i][0].item() for i in range(len(losses_valid))]
                    losses_recon_valid = [losses_valid[i][1].item() for i in range(len(losses_valid))]
                    losses_kl_valid = [losses_valid[i][2].item() for i in range(len(losses_valid))]

                    ensemble_loss_valid = torch.dot(torch.tensor(losses_total_valid, dtype=torch.float32), self.weights)
                    ensemble_loss_valid_recon = torch.dot(torch.tensor(losses_recon_valid, dtype=torch.float32), self.weights)
                    ensemble_loss_valid_kl = torch.dot(torch.tensor(losses_kl_valid, dtype=torch.float32), self.weights)

                    valid_total_loss.append(ensemble_loss_valid.item())
                    valid_recon_loss.append(ensemble_loss_valid_recon.item())
                    valid_kl_loss.append(ensemble_loss_valid_kl.item())

                ValidTotal_MeanBatch = np.mean(valid_total_loss)
                ValidRecon_MeanBatch = np.mean(valid_recon_loss)
                ValidKL_MeanBatch = np.mean(valid_kl_loss)

                self.ValidTotal_MeanBatch_Epochs.append(ValidTotal_MeanBatch)
                self.ValidRecon_MeanBatch_Epochs.append(ValidRecon_MeanBatch)
                self.ValidKL_MeanBatch_Epochs.append(ValidKL_MeanBatch)

            print(f"\nEpoch   Training   Validation    Validation MSE\n"
                  f"{epoch}   {TrainTotal_MeanBatch}  {ValidTotal_MeanBatch}  {ValidRecon_MeanBatch}\n"
                  f"====================================================")

            if ValidTotal_MeanBatch < best_val_loss:
                best_val_loss = ValidTotal_MeanBatch
                best_model_weights = self.model.state_dict()

            pbar.update()
        pbar.close()
        print("Done training!")
        if best_model_weights:
            self.model.load_state_dict(best_model_weights)

        return self.weights

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

















