import torch.nn as nn
import torch


class FNN_latent(nn.Module):

    def __init__(self, num_input, num_output, depth, width, relu=False):
        """
        Class that implements the fully connected neural network
        It is used to learn the mapping from the parameter space to the state space
        :param num_input: The number of input nodes
        :param num_output: The number of output nodes
        :param depth: number of hidden layers
        :param width: number of nodes in each layer
        """
        super().__init__()
        # Number of hidden layers of the network
        self.Depth = depth
        # Number of nodes in each hidden layer
        self.Width = width
        # Number of input nodes for the network
        self.NumInput = num_input
        # Number of output nodes for the network
        self.NumOutput = num_output

        self.relu = relu

        layers = []
        # Start with the input layer
        layers.append(nn.Linear(in_features=self.NumInput, out_features=self.Width))
        layers.append(nn.ReLU())

        for i in range(self.Depth):
            # Add hidden layers
            layers.append(nn.Linear(in_features=self.Width, out_features=self.Width))
            layers.append(nn.ReLU())

        # output layer
        layers.append(nn.Linear(in_features=self.Width, out_features=self.NumOutput))

        # try with relu on the output layer --> enforce non negative output for the logvar

        self.fnn_stack = nn.Sequential(*layers)
        # Add bias to enhance performance in unseen parameters
        self.b = torch.nn.parameter.Parameter(torch.zeros(self.NumOutput))

    def forward(self, x):
        fnn_output = self.fnn_stack(x)

        if self.relu:
            fnn_output = torch.relu(fnn_output)
            return fnn_output

        else:

            fnn_output += self.b
            return fnn_output


class VAE(nn.Module):
    def __init__(self, encoder, decoder, to_latent_params=None, to_decoder_params=None,
                 encode_metadata=False, enc_metadata_params=None, decoder_type="FC"):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mlp_to_latent = FNN_latent(*to_latent_params)
        self.mlp_to_decoder = FNN_latent(*to_decoder_params)
        self.encode_metadata = encode_metadata
        self.enc_metadata_params = enc_metadata_params
        self.decoder_type = decoder_type

    def forward(self, xyz, x_metadata):

        if self.encode_metadata:
            ### Encode the metadata / confounders using another MLP
            self.x_encoder = FNN_latent(*self.enc_metadata_params)
            self.x_metadata = self.x_encoder(x_metadata)

        y_encoded = self.encoder(xyz)

        y_to_latent = torch.cat((y_encoded, x_metadata), dim=1)

        latent_representation = self.mlp_to_latent(y_to_latent)

        mu = latent_representation[:, :int(latent_representation.shape[1]/2)]
        logvar = latent_representation[:, int(latent_representation.shape[1]/2):]

        z_sample = self.reparameterize(mu, logvar)

        self.z_dimensionality = z_sample.shape[1]

        ### Always employ mlp to give the input to the decoder
        y_to_decoder = torch.cat((z_sample, x_metadata), dim=1)
        y_to_decoder = self.mlp_to_decoder(y_to_decoder)

        if self.decoder_type == "PCN":

            self.xyz = xyz  ### To use it for the decoder --> does not actually influence the results
            recon_coarse, recon_dense = self.decoder(xyz=xyz, latent_vector=y_to_decoder)

            return recon_coarse, recon_dense, mu, logvar

        else:

            recon_coarse = None
            recon_dense = self.decoder(y_to_decoder)

            return recon_coarse, recon_dense, mu, logvar

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)

        return sample

    def generate(self, x_metadata, num_samples):

        x_metadata = torch.tensor(x_metadata.reshape(-1, 1), dtype=torch.float32)
        if self.encode_metadata:
            x_metadata = self.x_encoder(x_metadata)
        x_metadata_generate = torch.tile(x_metadata, (1, num_samples)).T

        ### generate (num_samples) number of random samples from the prior latent distribution
        ### prior is assumed to be a standard multivariate Gaussian distribution
        mean_prior = torch.zeros(self.z_dimensionality)
        cov_prior = torch.eye(self.z_dimensionality)

        distr_prior = torch.distributions.MultivariateNormal(mean_prior, cov_prior)
        z_samples = distr_prior.sample((num_samples,))

        y_to_decoder = torch.cat((z_samples, x_metadata_generate), dim=1)
        y_to_decoder = self.mlp_to_decoder(y_to_decoder)

        if self.decoder_type == "PCN":
            self.xyz = self.xyz[:num_samples, :, :] ### again, this does not have any functional purposes

            gen_shapes = self.decoder(self.xyz, y_to_decoder)[1]

        else:
            gen_shapes = self.decoder(y_to_decoder)

        return gen_shapes











