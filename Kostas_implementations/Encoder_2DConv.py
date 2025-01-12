import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class MLP2LatentDim(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(in_features=512 * 6 * 6, out_features=1000)
        self.layer2 = nn.Linear(in_features=1000, out_features=256)

        self.layer3 = nn.Linear(in_features=256, out_features=32)
        self.layer4 = nn.Linear(in_features=256, out_features=32)

        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(256)

        # self.bn3 = nn.BatchNorm1d(128)
        # self.bn4 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = F.gelu(self.bn1(self.layer1(x)))
        x = F.gelu(self.bn2(self.layer2(x)))

        ### TODO: possibly have 2 different layers here, distinct layers for mu and var (disentangle roles)
        mu = self.layer3(x)
        logvar = self.layer4(x)

        return mu, logvar


class MLP2DecoderDim(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(in_features=32, out_features=256)
        self.layer2 = nn.Linear(in_features=256, out_features=1000)
        self.layer3 = nn.Linear(in_features=1000, out_features=512 * 6 * 6)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(1000)
        self.bn3 = nn.BatchNorm1d(512 * 6 * 6)

    def forward(self, x):
        x = F.gelu(self.bn1(self.layer1(x)))
        x = F.gelu(self.bn2(self.layer2(x)))

        decoder_input = F.gelu(self.bn3(self.layer3(x)))

        return decoder_input


class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(128, 512, kernel_size=3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)
        self.bn7 = nn.BatchNorm2d(512)

    def forward(self, x):
        x_toskip = F.gelu(self.bn1(self.conv1(x)))

        x = F.gelu(self.bn2(self.conv2(x_toskip)))
        x = F.gelu(self.bn3(self.conv3(x)+x_toskip))
        x_toskip = F.gelu(self.bn4(self.conv4(x)))
        x = F.gelu(self.bn5(self.conv5(x_toskip)))
        x = F.gelu(self.bn6(self.conv6(x)+x_toskip))
        x = F.gelu(self.bn7(self.conv7(x)))

        encoded = x

        ### So now the output of the encoder will have the dimensionality of the latent space

        return encoded


class ConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.deconv1 = nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv6 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv7 = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=1, output_padding=1)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)

    def forward(self, x):
        x_toskip = F.gelu(self.bn1(self.deconv1(x)))
        x = F.gelu(self.bn2(self.deconv2(x_toskip)))
        x = F.gelu(self.bn3(self.deconv3(x)+x_toskip))
        x_toskip = F.gelu(self.bn4(self.deconv4(x)))
        x = F.gelu(self.bn5(self.deconv5(x_toskip)))
        x = F.gelu(self.bn6(self.deconv6(x)+x_toskip))
        decoded = self.deconv7(x)

        return decoded


class CVAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = ConvEncoder()
        self.decoder = ConvDecoder()
        self.to_decoder_input = MLP2DecoderDim()
        self.to_latent_mlp = MLP2LatentDim()

    def forward(self, x):
        batch_size = x.shape[0]

        y_encoded = self.encoder(x)

        ### Flatten the output of the encoder
        ### Fully-connected layers to reach the latent dimension
        x = y_encoded.view(y_encoded.size(0), -1)
        encoded_mu, encoded_logvar = self.to_latent_mlp(x)

        mu = encoded_mu
        logvar = encoded_logvar

        z_sample = self.reparameterize(mu, logvar)

        y_to_decoder = self.to_decoder_input(z_sample).reshape((batch_size, 512, 6, 6))

        reconstruction = self.decoder(y_to_decoder)

        return reconstruction, mu, logvar

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)

        return sample

    # def generate(self, x_metadata, num_samples):
    #     # x_metadata = torch.tensor(x_metadata.reshape(-1, 1), dtype=torch.float32)
    #     # x_metadata_generate = torch.tile(x_metadata, (1, num_samples)).T
    #
    #     ### generate (num_samples) number of random samples from the prior latent distribution
    #     ### prior is assumed to be a standard multivariate Gaussian distribution
    #     self.decoder.eval()
    #     with torch.no_grad():
    #         z_samples = torch.randn(num_samples, 32)
    #
    #         # y_to_decoder = torch.cat((z_samples, x_metadata_generate), dim=1)
    #         y_to_decoder = self.to_decoder_input(z_samples).reshape((num_samples, 512, 6, 6))
    #
    #         gen_shapes = self.decoder(y_to_decoder)
    #
    #     return gen_shapes


class CVAETrainer:
    def __init__(self, model, optimizer, epochs, train_loader, valid_loader):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.mean_loss_train_epochs = []
        self.mean_recon_train_epochs = []
        self.mean_kl_train_epochs = []

        self.mean_loss_valid_epochs = []
        self.mean_recon_valid_epochs = []
        self.mean_kl_valid_epochs = []

    @staticmethod
    def loss_function(beta_value, prediction, reference, logvar, mean):
        ### compare prediction with reconstruction "x"
        loss_reconstruct = torch.nn.MSELoss()
        loss_recon_ = loss_reconstruct(prediction, reference)
        loss_kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
        loss_kl = torch.mean(loss_kl)

        loss_func = loss_recon_ + beta_value * loss_kl

        return loss_func, loss_recon_, beta_value * loss_kl

    def training(self):
        best_val_loss = float('inf')
        best_model_weights = None
        pbar = tqdm(total=self.epochs, desc="Epochs training...")
        for epoch in range(self.epochs):

            ### Training phase
            ### Annealing of beta term, used in the loss of the beta-VAE
            if epoch >= 750:
                beta = 2
            elif epoch >= 600:
                beta = 0.8
            elif epoch >= 500:
                beta = 0.4
            elif epoch >= 400:
                beta = 0.1
            else:
                beta = 0

            self.model.train(True)

            total_loss_train = []
            recon_loss_train = []
            kl_loss_train = []
            for x_train, _ in self.train_loader:
                self.optimizer.zero_grad()

                ### Forward propagation. Inputs to VAE are the batch of point clouds and the patient metadata
                prediction, mu, log_var = self.model(x_train)

                loss_train, loss_recon, kl_loss = self.loss_function(beta_value=beta, prediction=prediction,
                                                                     logvar=log_var, mean=mu, reference=x_train)

                ### Backprop
                loss_train.backward()
                self.optimizer.step()

                total_loss_train.append(loss_train.item())

                recon_loss_train.append(loss_recon.detach().numpy())
                kl_loss_train.append(kl_loss.detach().numpy())

            mean_loss_train_epoch = np.mean(total_loss_train)
            mean_loss_recon_epoch = np.mean(recon_loss_train)
            mean_loss_kl_epoch = np.mean(kl_loss_train)

            self.mean_loss_train_epochs.append(mean_loss_train_epoch)
            self.mean_recon_train_epochs.append(mean_loss_recon_epoch)
            self.mean_kl_train_epochs.append(mean_loss_kl_epoch)

            ### Validation phase
            self.model.eval()
            total_loss_valid = []
            mse_losses_valid = []
            recon_loss_valid = []
            kl_loss_valid = []
            with torch.no_grad():
                for x_valid, _ in self.valid_loader:
                    prediction_valid, mu_valid, log_var_valid = self.model(x_valid)

                    loss_valid, loss_recon_valid, loss_kl_valid = self.loss_function(beta_value=beta,
                                                                                     prediction=prediction_valid,
                                                                                     logvar=log_var_valid, mean=mu_valid,
                                                                                     reference=x_valid)

                    total_loss_valid.append(loss_valid.item())
                    recon_loss_valid.append(loss_recon_valid.detach().numpy())
                    kl_loss_valid.append(loss_kl_valid.detach().numpy())

                    mse_losses_valid.append(loss_recon_valid.detach().numpy())

            mean_loss_valid_epoch = np.mean(total_loss_valid)
            mean_recon_valid_epoch = np.mean(recon_loss_valid)
            mean_kl_valid_epoch = np.mean(kl_loss_valid)

            self.mean_loss_valid_epochs.append(mean_loss_valid_epoch)
            self.mean_recon_valid_epochs.append(mean_recon_valid_epoch)
            self.mean_kl_valid_epochs.append(mean_kl_valid_epoch)

            mean_loss_mse_epoch_valid = np.mean(mse_losses_valid)

            # Keep track of the model that results to the minimum validation error
            # if mean_loss_valid_epoch < best_val_loss:
            #     best_val_loss = mean_loss_valid_epoch
            #     best_model_weights = self.model.state_dict()

            print(f"\nEpoch   Training   Validation    Validation MSE    Beta\n"
                  f"{epoch}   {mean_loss_train_epoch}  {mean_loss_valid_epoch}  {mean_loss_mse_epoch_valid}  {beta}\n"
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

        # if best_model_weights:
        #     self.model.load_state_dict(best_model_weights)

        return self.model

    def plot_losses(self):

        ### Plot the losses
        plt.figure()
        plt.semilogy(self.mean_loss_train_epochs, label='Total error')
        plt.semilogy(self.mean_recon_train_epochs, label='Reconstruction')
        plt.semilogy(self.mean_kl_train_epochs, label='KL')
        plt.title("Training")
        plt.xlabel("# Epochs")
        plt.ylabel("Loss function")
        plt.legend()
        plt.show()

        plt.figure()
        plt.semilogy(self.mean_loss_valid_epochs, label='Total error')
        plt.semilogy(self.mean_recon_valid_epochs, label='Reconstruction')
        plt.semilogy(self.mean_kl_valid_epochs, label='KL')
        plt.xlabel("# Epochs")
        plt.ylabel("Loss function")
        plt.legend()
        plt.show()
