import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


class Encoder(nn.Module):
    """
    3D convolutional encoder that maps momenta to a latent vector.

    This encoder applies a sequence of 3D convolutional layers followed by GELU
    activations. The resulting feature map is flattened and passed through a
    fully connected layer to obtain a latent representation.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent representation produced by the encoder.

    Attributes
    ----------
    conv1 : torch.nn.Conv3d
        First 3D convolutional layer. Maps the input from 3 channels to 32
        feature channels using a 1x1x1 kernel.

    conv2 : torch.nn.Conv3d
        Second 3D convolutional layer. Keeps 32 feature channels and uses a
        2x2x2 kernel.

    conv3 : torch.nn.Conv3d
        Third 3D convolutional layer. Maps the feature representation from
        32 channels to 64 channels using a 1x1x1 kernel.

    conv4 : torch.nn.Conv3d
        Fourth 3D convolutional layer. Maps the feature representation from
        64 channels to 128 channels using a 2x2x2 kernel.

    flatten : torch.nn.Flatten
        Flattens the final convolutional feature map into a one-dimensional
        feature vector per sample.

    fc_latent : torch.nn.Linear
        Fully connected layer that maps the flattened feature vector to the
        latent space.
    """
    def __init__(self, latent_dim):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=(1, 1, 1), stride=1).to('cuda')
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(2, 2, 2), stride=1).to('cuda')
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(1, 1, 1), stride=1).to('cuda')
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(2, 2, 2), stride=1).to('cuda')

        self.flatten = nn.Flatten().to('cuda')

        self.fc_latent = nn.Linear(128 * 6 * 7 * 8, latent_dim).to('cuda')


    def forward(self, x):
        """
                Forward pass of the encoder.

                Parameters
                ----------
                x : torch.Tensor
                    Input tensor with shape:

                    `(batch_size, 3, D, H, W)`

                    For the current fully connected layer, the expected spatial input
                    size is approximately `(8, 9, 10)`.

                Returns
                -------
                z_latent : torch.Tensor
                    Latent representation of the input tensor with shape:

                    `(batch_size, latent_dim)`
                """

        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = F.gelu(self.conv4(x))

        x = self.flatten(x)

        z_latent = self.fc_latent(x)

        return z_latent


class Decoder(nn.Module):
    """
        3D transposed-convolutional decoder that maps a latent vector back to the
        momenta space.

        This decoder performs the inverse operation of the corresponding encoder.
        It first maps the latent representation to a high-dimensional feature vector
        using a fully connected layer. The feature vector is then reshaped into a
        3D feature map and passed through a sequence of transposed 3D convolutional
        layers to reconstruct the momenta output.

        Parameters
        ----------
        latent_dim : int
            Dimensionality of the latent representation used as input to the decoder.

        Attributes
        ----------
        fc : torch.nn.Linear
            Fully connected layer that maps the latent vector to a flattened 3D
            feature representation of size `128 * 6 * 7 * 8`.

        deconv1 : torch.nn.ConvTranspose3d
            First transposed 3D convolutional layer. Maps the feature representation
            from 128 channels to 64 channels using a 2x2x2 kernel.

        deconv2 : torch.nn.ConvTranspose3d
            Second transposed 3D convolutional layer. Maps the feature representation
            from 64 channels to 32 channels using a 1x1x1 kernel.

        deconv3 : torch.nn.ConvTranspose3d
            Third transposed 3D convolutional layer. Keeps 32 feature channels and
            uses a 2x2x2 kernel.

        deconv4 : torch.nn.ConvTranspose3d
            Final transposed 3D convolutional layer. Maps the feature representation
            from 32 channels back to 3 output channels.
        """
    def __init__(self, latent_dim):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 128 * 6 * 7 * 8).to('cuda')

        self.deconv1 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=(2, 2, 2), stride=1).to('cuda')
        self.deconv2 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=(1, 1, 1), stride=1).to('cuda')
        self.deconv3 = nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=(2, 2, 2), stride=1).to('cuda')
        self.deconv4 = nn.ConvTranspose3d(in_channels=32, out_channels=3, kernel_size=(1, 1, 1), stride=1).to('cuda')


    def forward(self, z):
        """
                Forward pass of the decoder.

                Parameters
                ----------
                z : torch.Tensor
                    Latent input tensor with shape:

                    `(batch_size, latent_dim)`

                Returns
                -------
                x : torch.Tensor
                    Reconstructed volumetric tensor with shape:

                    `(batch_size, 3, 8, 9, 10)`

                    for the current architecture.
                """

        x = F.gelu(self.fc(z))

        x = x.view(-1, 128, 6, 7, 8)

        x = F.gelu(self.deconv1(x))
        x = F.gelu(self.deconv2(x))
        x = F.gelu(self.deconv3(x))

        x = self.deconv4(x)

        return x


class ConvAE(nn.Module):
    """
    3D convolutional autoencoder

    This class combines an encoder and a decoder into a complete autoencoder
    architecture. The encoder maps the input momenta to a lower-dimensional latent
    representation, while the decoder reconstructs the original momenta
    from this latent representation.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent representation produced by the encoder and
        used as input to the decoder.

    Attributes
    ----------
    encoder : Encoder
        Encoder network that maps the input tensor to a latent vector.

    decoder : Decoder
        Decoder network that maps the latent vector back to the reconstructed
        volumetric output
    """
    def __init__(self, latent_dim):
        super().__init__()

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):

        z_latent = self.encoder(x)
        out_shape = self.decoder(z_latent)

        return out_shape


class TrainerAE:
    """
    Trainer class for a convolutional autoencoder.

    This class handles the complete training and validation loop for an
    autoencoder model. It trains the model using MSE-based reconstruction error
    loss

    Parameters
    ----------
    model : torch.nn.Module
        Autoencoder model to be trained. The model is expected to take an input
        tensor and return a reconstructed tensor with the same shape.

    optimizer : torch.optim.Optimizer
        Optimizer used to update the model parameters.

    epochs : int
        Number of training epochs.

    train_loader : torch.utils.data.DataLoader
        DataLoader containing the training data.

    valid_loader : torch.utils.data.DataLoader
        DataLoader containing the validation data.

    latent_dim : int
        Dimensionality of the latent representation. This is stored for reference
        but is not directly used during training.

    Attributes
    ----------
    device : torch.device
        Device used for training. In the current implementation, this is fixed
        to CUDA.

    TrainTotal_MeanBatch_Epochs : list
        List storing the mean training loss for each epoch.

    ValidTotal_MeanBatch_Epochs : list
        List storing the mean validation loss for each epoch.

    latent_dimension : int
        Stored value of the latent dimensionality.
    """
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
        """
        Train the autoencoder model.

        This method performs the full training and validation loop. For each
        epoch

        Returns
        -------
        model : torch.nn.Module
            The trained model loaded with the weights corresponding to the lowest
            validation loss.
        """
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
