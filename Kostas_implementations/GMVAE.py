import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm


class Qy_x(nn.Module):
    """Conditional distribution q(y|x) represented by a neural network.

    Args:
        encoder (nn.Module): The encoder module used to process the input data.
        Perhaps a convolutional encoded is needed here
        enc_out_dim (int): The output dimension of the encoder module.
        k (int): Number of components in the Gaussian mixture prior.

    Attributes:
        h1 (nn.Module): The encoder module used to process the input data.
        qy_logit (nn.Linear): Linear layer for predicting the logit of q(y|x).
        qy (nn.Softmax): Softmax activation function for q(y|x).

    """
    def __init__(self, encoder, enc_out_dim, k):
        super(Qy_x, self).__init__()
        self.h1 = encoder
        self.qy_logit = nn.Linear(enc_out_dim, k)
        self.qy = nn.Softmax(dim=1)

    def forward(self, x):
        """Perform the forward pass for q(y|x).

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            tuple: A tuple containing the logit and softmax outputs of q(y|x).
        """
        h1 = self.h1(x)
        qy_logit = self.qy_logit(h1)
        qy_prob = self.qy(qy_logit)
        return qy_logit, qy_prob


class Qz_xy(nn.Module):
    """Conditional distribution q(z|x, y) represented by a neural network.

    Args:
        k (int): Number of components in the Gaussian mixture prior.
        encoder (nn.Module): The encoder module used to process the input data.
        Convolutional encoder
        enc_out_dim (int): The output dimension of the encoder module.
        hidden_size (int): Number of units in the hidden layer.
        latent_dim (int): Dimensionality of the latent space.

    Attributes:
        h1 (nn.Module): The encoder module used to process the input data.
        h2 (nn.Sequential): The hidden layers of the neural network.
        z_mean (nn.Linear): Linear layer for predicting the mean of q(z|x, y).
        zlogvar (nn.Linear): Linear layer for predicting
            the log variance of q(z|x, y).

    """
    def __init__(self, k, encoder, enc_out_dim, hidden_size, latent_dim):
        super(Qz_xy, self).__init__()
        self.h1 = encoder
        self.h2 = nn.Sequential(
            nn.Linear(enc_out_dim + k, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.z_mean = nn.Linear(hidden_size, latent_dim)
        self.zlogvar = nn.Linear(hidden_size, latent_dim)

    def gaussian_sample(self, z_mean, z_logvar):
        z_std = torch.sqrt(torch.exp(z_logvar))

        eps = torch.randn_like(z_std)
        z = z_mean + eps*z_std

        return z

    def forward(self, x, y):
        """Perform the forward pass for q(z|x, y).

        Args:
            x (torch.Tensor): Input data tensor.
            y (torch.Tensor): One-hot encoded tensor representing
                the class labels.

        Returns:
            tuple: A tuple containing the latent variables, mean,
                and log variance of q(z|x, y).
        """
        h1 = self.h1(x)
        xy = torch.cat((h1, y), dim=1)
        h2 = self.h2(xy)
        # q(z|x, y)
        z_mean = self.z_mean(h2)
        zlogvar = self.zlogvar(h2)
        z = self.gaussian_sample(z_mean, zlogvar)
        return z, z_mean, zlogvar


class Px_z(nn.Module):
    """Conditional distribution p(x|z) represented by a neural network.

    Args:
        decoder (nn.Module): The decoder module used to reconstruct the data.
        k (int): Number of components in the Gaussian mixture prior.

    Attributes:
        decoder (nn.Module): The decoder module used to reconstruct the data.
        decoder_hidden (int): Number of units in the hidden layer
            of the decoder.
        latent_dim (int): Dimensionality of the latent space.
        z_mean (nn.Linear): Linear layer for predicting the mean of p(z|y).
        zlogvar (nn.Linear): Linear layer for predicting the log variance
            of p(z|y).

    """
    def __init__(self, decoder, k):
        super(Px_z, self).__init__()
        self.decoder = decoder
        self.decoder_hidden = self.decoder.hidden_size
        self.latent_dim = self.decoder.latent_dim
        self.z_mean = nn.Linear(k, self.latent_dim)
        self.zlogvar = nn.Linear(k, self.latent_dim)

    def forward(self, z, y):
        """Perform the forward pass for p(x|z) and p(z|y).

        Args:
            z (torch.Tensor): Latent variable tensor.
            y (torch.Tensor): One-hot encoded tensor representing
                the class labels.

        Returns:
            tuple: A tuple containing the prior mean, prior log variance,
                and reconstructed data.
        """
        # p(z|y)
        z_mean = self.z_mean(y)
        zlogvar = self.zlogvar(y)

        # p(x|z)
        x_hat = self.decoder(z)
        return z_mean, zlogvar, x_hat


class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=52, kernel_size=3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(52)

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))

        x = self.flatten(x)

        return x


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(ConvDecoder, self).__init__()

        self.fc = nn.Linear(latent_dim, 52 * 12 * 12)

        self.deconv1 = nn.ConvTranspose2d(in_channels=52, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=7, stride=2, padding=1, output_padding=1)

        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, z):
        x = F.gelu(self.fc(z))

        x = x.view(-1, 52, 12, 12)

        x = F.gelu(self.bn1(self.deconv1(x)))
        x = self.deconv2(x)

        return x


class GMVAE(nn.Module):
    """GMVAE model for Variational Autoencoders with Gaussian Mixture Prior.

    Args:
        k (int): Number of components in the Gaussian mixture prior.
        Qy_x_net (nn.Module): Neural network module representing q(y|x).
        Qz_xy_net (nn.Module): Neural network module representing q(z|x, y).
        Px_z_net (nn.Module): Neural network module representing p(x|z).

    Attributes:
        k (int): Number of components in the Gaussian mixture prior.
        qy_x (nn.Module): Neural network module representing q(y|x).
        qz_xy (nn.Module): Neural network module representing q(z|x, y).
        px_z (nn.Module): Neural network module representing p(x|z).

    """

    def __init__(self, k, Qy_x_net, Qz_xy_net, Px_z_net):
        super(GMVAE, self).__init__()
        self.k = k
        self.qy_x = Qy_x_net
        self.qz_xy = Qz_xy_net
        self.px_z = Px_z_net

    def infer(self, x):
        """Perform inference for a given input data.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            dict: A dictionary containing inferred output values, including y_hat, z, and x_hat.
        """
        k = self.k
        batch_size = x.shape[0]
        _, qy = self.qy_x(x)
        y_hat = torch.argmax(qy, dim=-1)

        # Create tensor with 1s at specified indices
        y_ = torch.zeros(batch_size, k)
        y_ = torch.scatter(y_, 1, y_hat.unsqueeze(1), 1)
        z_hat, *_ = self.qz_xy(x, y_)
        *_, x_hat = self.px_z(z_hat, y_)
        out_infer = {
            "y": y_hat,
            "z": z_hat,
            "x_hat": x_hat
        }
        return out_infer

    def forward(self, x):
        """Perform forward pass through the GMVAE model.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            tuple: A tuple containing two dictionaries. The first dictionary contains
            training-related outputs such as z, zm, zv, zm_prior, zv_prior, qy_logit,
            qy, and px.
            The second dictionary contains inference-related outputs such as
            y_hat, z_hat, x_hat, and qy.
        """
        k = self.k
        batch_size = x.shape[0]
        y_ = torch.zeros([batch_size, k]).to(x.device)
        qy_logit, qy = self.qy_x(x)
        z, zm, zv, zm_prior, zv_prior, px = [[None] * k for i in range(6)]
        for i in range(k):
            y = y_ + torch.eye(k).to(x.device)[i]
            z[i], zm[i], zv[i] = self.qz_xy(x, y)
            zm_prior[i], zv_prior[i], px[i] = self.px_z(z[i], y)

        # Inference for x_hat:
        with torch.no_grad():
            y_hat = torch.argmax(qy, dim=-1)
            y_temp = torch.zeros(batch_size, k)
            y_temp = torch.scatter(y_, 1, y_hat.unsqueeze(1), 1)
            z_hat, *_ = self.qz_xy(x, y_temp)
            *_, x_hat = self.px_z(z_hat, y_temp)

        out_train = {
            "z": z,
            "zm": zm,
            "zv": zv,
            "zm_prior": zm_prior,
            "zv_prior": zv_prior,
            "qy_logit": qy_logit,
            "qy": qy,
            "px": px
        }

        out_infer = {
            "y": y_hat,
            "z": z_hat,
            "x_hat": x_hat,
            "qy": qy
        }
        return out_train, out_infer


class MSE:
    """
    Mean Squared Error (MSE) loss function.

    Calculates the mean squared error between the input and the target.

    Args:
        x (torch.Tensor): Input tensor.
        x_hat (torch.Tensor): Target tensor.

    Returns:
        torch.Tensor: Computed loss tensor.
    """

    def __call__(self, x, x_hat):
        batch_size = x.shape[0]
        loss = nn.MSELoss(reduction='none')(x, x_hat)
        loss = loss.view(batch_size, -1).sum(axis=1)
        return loss


class TotalLoss:
    """
    generative process:
    p_theta(x, y, z) = p_theta(x|z) p_theta(z|y) p(y)
    y ~ Cat(y|1/k)
    z|y ~ N(z|mu_z_theta(y), sigma^2*z_theta(y))
    x|z ~ B(x|mu_x_theta(z))

    The goal of GMVAE is to estimate the posterior
    distribution p(z, y|x),
    which is usually difficult to compute directly.
    Instead, a factorized posterior,
    known as the inference model,
    is commonly used as an approximation:

    q_phi(z, y|x) = q_phi(z|x, y) q_phi(y|x)
    y|x ~ Cat(y|pi_phi(x))
    z|x, y ~ N(z|mu_z_phi(x, y), sigma^2z_phi(x, y))

    ELBO = -KL(q_phi(z|x, y) || p_theta(z|y))
            - KL(q_phi(y|x) || p(y)) + Eq_phi(z|x,y) [log p_theta(x|z)]

    """

    def __init__(self, k, recon_loss=MSE()):
        self.k = k
        self.recon_loss = recon_loss

    def negative_entropy_from_logit(self, qy, qy_logit):
        """
        Computes:
        ???
        ++++++++++++++++++++++++++++++++++++++++++++
        H(q, q) = - ∑q*log q
        H(q, q_logit) = - ∑q*log p(q_logit)
        p(q_logit) = softmax(qy_logit)
        H(q, q_logit) = - ∑q*log softmax(qy_logit)
        ++++++++++++++++++++++++++++++++++++++++++++
        """
        nent = torch.sum(qy * torch.nn.LogSoftmax(1)(qy_logit), 1)
        return nent

    def log_normal(self, x, mu, var, eps=0., axis=-1):
        """
        Calculate the element-wise log probability of a normal distribution.

        The function computes the logarithm of the probability density function of a
        normal distribution given the input tensor `x`, mean tensor `mu`, and variance
        tensor `var`. It is assumed that `x`, `mu`, and `var` have the same shape.

        Args:
            x (torch.Tensor): Input tensor.
            mu (torch.Tensor): Mean tensor of the normal distribution.
            var (torch.Tensor): Variance tensor of the normal distribution.
            eps (float, optional): A small value added to the variance to avoid numerical instability.
                Defaults to 0.0.
            axis (int, optional): The axis along which the log probabilities are summed.
                Defaults to -1.

        Returns:
            torch.Tensor: The computed log probability of the normal distribution.
        """
        if eps > 0.0:
            var = torch.add(var, eps)
        return -0.5 * torch.sum(np.log(2 * math.pi) + torch.log(var) + torch.pow(x - mu, 2) / var, axis)

    def _loss_per_class(self, x, x_hat, z, zm, zv, zm_prior, zv_prior):
        loss_px_i = self.recon_loss(x, x_hat)
        loss_px_i += self.log_normal(z, zm, zv) - self.log_normal(z, zm_prior, zv_prior)
        return loss_px_i - np.log(1 / self.k)

    def __call__(self, x, output_dict):
        qy = output_dict["qy"]
        qy_logit = output_dict["qy_logit"]
        px = output_dict["px"]
        z = output_dict["z"]
        zm = output_dict["zm"]
        zv = output_dict["zv"]
        zm_prior = output_dict["zm_prior"]
        zv_prior = output_dict["zv_prior"]
        loss_qy = self.negative_entropy_from_logit(qy, qy_logit)
        losses_i = []
        for i in range(self.k):
            losses_i.append(
                self._loss_per_class(
                    x, px[i], z[i], zm[i], torch.exp(zv[i]), zm_prior[i], torch.exp(zv_prior[i]))
            )
        loss = torch.stack([loss_qy] + [qy[:, i] * losses_i[i] for i in range(self.k)]).sum(0)
        # Alternative way to calculate loss:
        # torch.sum(torch.mul(torch.stack(losses_i), torch.transpose(qy, 1, 0)), dim=0)
        out_dict = {"cond_entropy": loss_qy.sum(), "total_loss": loss.sum()}
        return out_dict


class TrainerGMVAE:
    def __init__(self, model, optimizer, epochs, train_loader, valid_loader, criterion):

        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion

        self.TrainTotal_MeanBatch_Epochs = []

        self.ValidTotal_MeanBatch_Epochs = []

    def training(self):

        pbar = tqdm(total=self.epochs, desc="Epochs training...")
        for epoch in range(self.epochs):

            self.model.train()
            train_total_loss = []
            for x_train, _ in self.train_loader:
                self.optimizer.zero_grad()

                ### Forward propagation. Inputs to VAE are the batch of point clouds and the patient metadata
                out_train, out_infer = self.model(x_train)
                loss = self.criterion(x_train, out_train)

                loss['total_loss'].backward()
                self.optimizer.step()
                train_total_loss.append(loss['total_loss'].item())

            TrainTotal_MeanBatch = np.mean(train_total_loss)

            self.TrainTotal_MeanBatch_Epochs.append(TrainTotal_MeanBatch)

            print(f"\nEpoch   Training   Validation    Validation MSE    Beta\n"
                  f"{epoch}   {TrainTotal_MeanBatch}  \n"
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
        plt.title("Training")
        plt.xlabel("# Epochs")
        plt.ylabel("Loss function")
        plt.legend()
        plt.show()

        plt.figure()
        plt.semilogy(self.ValidTotal_MeanBatch_Epochs, label='Total error')
        plt.xlabel("# Epochs")
        plt.ylabel("Loss function")
        plt.legend()
        plt.show()

