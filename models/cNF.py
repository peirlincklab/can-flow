import torch
import torch.nn as nn
import numpy as np
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm


class ActNorm(nn.Module):
    def __init__(self, num_features, device, init=True):
        """
        ActNorm layer for an MLP-based normalizing flow.

        Args:
            num_features (int): Number of features in the input.
            init (bool): Whether to initialize using the first batch.
        """
        super().__init__()
        self.num_features = num_features
        self.initialized = not init

        self.s = nn.Parameter(torch.ones(1, num_features).to(device))
        self.b = nn.Parameter(torch.zeros(1, num_features).to(device))

    def forward(self, x):
        """
        Forward transformation: y = s * x + b
        """
        if not self.initialized:

            with torch.no_grad():
                mean = x.mean(dim=0, keepdim=True)
                std = x.std(dim=0, keepdim=True)

                # Initialize parameters
                self.b.data.copy_(-mean)
                self.s.data.copy_(1 / (std+ 1e-8))
                self.initialized = True  # Mark as initialized


        y = self.s * x + self.b
        log_det = torch.sum(torch.log(torch.abs(self.s))).expand(x.shape[0])

        return y, log_det

    def reverse(self, y):
        """
        Inverse transformation: x = (y - b) / s
        """
        x = (y - self.b) / self.s

        return x



class InvertibleLinearTransform(nn.Module):
    def __init__(self, dim, device):
        """
        Invertible linear transform layer
        :param dim: input dimension
        :param device: device
        """
        super().__init__()

        self.device = device

        np_w = scipy.linalg.qr(np.random.randn(dim, dim))[0].astype(np.float64)

        ### LU decomposition
        np_p, np_l, np_u = scipy.linalg.lu(np_w, permute_l=False)

        np_s = np.diag(np_u)
        np_sign_s = np.sign(np_s)
        np_log_s = np.log(np.abs(np_s))
        np_u = np.triu(np_u, k=1)

        ### buffer parameters
        self.register_buffer('P', torch.from_numpy(np_p).float().to(device))
        self.register_buffer('sign_s', torch.from_numpy(np_sign_s).float().to(device))
        self.l_param = nn.Parameter(torch.from_numpy(np_l).float().to(self.device))
        self.u_param = nn.Parameter(torch.from_numpy(np_u).float().to(self.device))
        self.log_s = nn.Parameter(torch.from_numpy(np_log_s).float().to(self.device))

        ### masking
        mask = np.tril(np.ones((dim, dim), dtype=np.float32), -1)
        self.register_buffer('l_mask', torch.from_numpy(mask).to(device))
        self.register_buffer('u_mask', torch.from_numpy(mask.T).to(device))

    def forward(self, x):
        L = self.l_param * self.l_mask + torch.eye(x.shape[1], device=self.device)
        U = self.u_param * self.u_mask + torch.diag(self.sign_s * torch.exp(self.log_s))

        W = self.P @ L @ U

        logdet = torch.sum(self.log_s).expand(x.shape[0])

        return x @ W.T, logdet

    def reverse(self, y):
        L = self.l_param * self.l_mask + torch.eye(y.shape[1], device=self.device)
        U = self.u_param * self.u_mask + torch.diag(self.sign_s * torch.exp(self.log_s))

        W = self.P @ L @ U

        W_inv = torch.inverse(W)

        return y @ W_inv.T


class AffineInjector(nn.Module):
    def __init__(self, out_dim, activation):
        """
        Affine injector layer
        It is used to further inform the normalizing flow on the metadata conditioning
        :param out_dim: output dimension
        :param activation: activation function to be used in the s, t networks
        """
        super().__init__()

        self.aff_inj_s = nn.Sequential(
            nn.Linear(4, 20),
            activation,
            nn.Linear(20, 60),
            activation,
            nn.Linear(60, out_dim)
        ).to('cuda')

        self.aff_inj_t = nn.Sequential(
            nn.Linear(4, 20),
            activation,
            nn.Linear(20, 60),
            activation,
            nn.Linear(60, out_dim)
        ).to('cuda')

        def init_weights_zero(m):
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

        self.aff_inj_s.apply(init_weights_zero)
        self.aff_inj_t.apply(init_weights_zero)


    def forward(self, x, x_conf, compute_likelihoods=False):
        s = self.aff_inj_s(x_conf)
        t = self.aff_inj_t(x_conf)

        x_new = x * torch.exp(s) + t

        logdet = torch.sum(torch.log(torch.abs(torch.exp(s))), dim=1)

        if not compute_likelihoods:
            return x_new, logdet
        else:
            return None, logdet

    def reverse(self, x, x_conf):

        s = self.aff_inj_s(x_conf)
        t = self.aff_inj_t(x_conf)

        x_new = (x - t) * torch.exp(-s)

        return x_new



class AffineCoupling(nn.Module):
    def __init__(self, in_dim, out_dim, h_x_conf_dim, device, activation):
        """
        Affine coupling layer
        :param in_dim: input dimension
        :param out_dim: output dimension
        :param h_x_conf_dim: dimension of metadata embedding
        :param device: device
        :param activation: activation function
        """
        super().__init__()

        self.coupling_s = nn.Sequential(
            nn.Linear(int(in_dim/2)+h_x_conf_dim, out_dim),
            activation,
            nn.Linear(out_dim, out_dim),
            activation,
            nn.Linear(out_dim, int(in_dim/2)),
        )

        self.coupling_t = nn.Sequential(
            nn.Linear(int(in_dim/2)+h_x_conf_dim, out_dim),
            activation,
            nn.Linear(out_dim, out_dim),
            activation,
            nn.Linear(out_dim, int(in_dim/2))
        )

        self.coupling_s = self.coupling_s.to(device)
        self.coupling_t = self.coupling_t.to(device)

        def init_weights_zero(m):
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

        self.coupling_t.apply(init_weights_zero)
        self.coupling_s.apply(init_weights_zero)

    def forward(self, x, h_x_conf, compute_likelihoods=False):

        x_a, x_b = torch.chunk(x, 2, dim=1)

        x_a_cat = torch.cat((x_a, h_x_conf), dim=1)

        s = self.coupling_s(x_a_cat)
        t = self.coupling_t(x_a_cat)

        z_a = x_a
        z_b = x_b * torch.exp(s) + t

        z_out = torch.cat((z_a, z_b), dim=1)

        logdet = torch.sum(torch.log(torch.abs(torch.exp(s))), dim=1)

        if not compute_likelihoods:
            return z_out, logdet
        else:
            return None, logdet


    def reverse(self, z, h_x_conf):
        z_a, z_b = torch.chunk(z, 2, dim=1)

        z_a_cat = torch.cat((z_a, h_x_conf), dim=1)
        s = self.coupling_s(z_a_cat)
        t = self.coupling_t(z_a_cat)

        x_a = z_a
        x_b = (z_b - t) * torch.exp(-s)

        x_out = torch.cat((x_a, x_b), dim=1)

        return x_out


class Block(nn.Module):
    def __init__(self, in_dim, out_dim, h_x_conf_dim, activation, device):
        """
        Implements a single block of the normalizing flow architecture, with the several above layers
        :param in_dim: input dimension
        :param out_dim: output dimension
        :param h_x_conf_dim: dimension of metadata embedding
        :param activation: activation function
        :param device: device
        """
        super().__init__()

        self.actnorm = ActNorm(num_features=in_dim, device=device)
        self.lin_transform = InvertibleLinearTransform(dim=in_dim, device=device)
        self.coupling_layer = AffineCoupling(in_dim=in_dim, out_dim=out_dim, h_x_conf_dim=h_x_conf_dim,
                                             device=device, activation=activation)

        self.affine_injector = AffineInjector(out_dim=in_dim, activation=activation)


    def forward(self, x, h_x_conf, x_conf):

        out, logdet_actnorm = self.actnorm(x)
        out, logdet_w = self.lin_transform(out)
        out, logdet_affinj = self.affine_injector(out, x_conf)
        out, logdet_coupl = self.coupling_layer(out, h_x_conf)

        logdet = logdet_actnorm + logdet_w + logdet_affinj + logdet_coupl

        return out, logdet

    def reverse(self, x, h_x_conf, x_conf):

        x = self.coupling_layer.reverse(x, h_x_conf)
        x = self.affine_injector.reverse(x, x_conf)
        x = self.lin_transform.reverse(x)
        x = self.actnorm.reverse(x)

        return x


class FlowHeart(nn.Module):
    def __init__(self, in_dim, out_dim, n_flow, in_dim_conf, out_dim_conf, device, activations=None):
        """
        Implements the whole normalizing flow generative model, consisting of several blocks
        :param in_dim: input dimension
        :param out_dim: output dimension
        :param n_flow: number of blocks
        :param in_dim_conf: input dimension of metadata conditioning
        :param out_dim_conf: output dimension of metadata embedding
        :param device: device
        :param activations: activation functions
        """
        super().__init__()

        if activations is not None:
            self.activation = activations
        else:
            self.activation = nn.LeakyReLU()

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Block(in_dim=in_dim, out_dim=out_dim, h_x_conf_dim=out_dim_conf,
                                    activation=self.activation, device=device))


        self.conf_embedding = nn.Sequential(
            nn.Linear(in_dim_conf, out_dim_conf),
            self.activation,
            nn.Linear(out_dim_conf, out_dim_conf)
        )

        self.mean_prior = nn.Sequential(
            nn.Linear(out_dim_conf, int(in_dim/2)),
            self.activation,
            nn.Linear(int(in_dim/2), in_dim)
        )

        self.var_prior = nn.Sequential(
            nn.Linear(out_dim_conf, int(in_dim / 2)),
            self.activation,
            nn.Linear(int(in_dim / 2), in_dim),
            nn.Softplus()
        )

        self.conf_embedding = self.conf_embedding.to(device)
        self.mean_prior = self.mean_prior.to(device)
        self.var_prior = self.var_prior.to(device)


    def forward(self, x, x_conf, compute_likelihoods):

        h_x_conf = self.conf_embedding(x_conf)

        mean = self.mean_prior(h_x_conf)
        var = self.var_prior(h_x_conf)
        cov = torch.diag_embed(var)

        prior = torch.distributions.MultivariateNormal(mean, cov)

        logdet = torch.zeros(x.shape[0], device=x.device)

        for i, flow in enumerate(self.flows):
            x, det = flow(x, h_x_conf, x_conf)
            logdet += det

        # compute prior.log_prob(x) -> per-sample vector
        log_p = prior.log_prob(x)
        p_total = log_p + logdet  # per-sample --> This is not mean yet

        if not compute_likelihoods:
            return x, log_p, logdet
        else:
            return p_total


    def reverse(self, x_conf):

        h_x_conf = self.conf_embedding(x_conf)

        mean = self.mean_prior(h_x_conf)
        var = self.var_prior(h_x_conf)
        cov = torch.diag_embed(var)

        prior = torch.distributions.MultivariateNormal(mean, cov)

        outs = []
        ### Sample from the prior
        x = prior.sample()
        # x = x.reshape(x_conf.shape[0], mean.shape[1])

        outs.append(x)
        for i, flow in enumerate(self.flows[::-1]):
            x = flow.reverse(x, h_x_conf, x_conf)
            outs.append(x)

        return x, outs









class TrainerNF:
    def __init__(self, model, optimizer, epochs, train_loader, valid_loader, latent_dim):

        self.device = torch.device('cuda')

        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.TrainTotal_MeanBatch_Epochs = []
        self.Trainlog_p_MeanBatch_Epochs = []
        self.Trainlogdet_MeanBatch_Epochs = []

        self.ValidTotal_MeanBatch_Epochs = []
        self.Validlog_p_MeanBatch_Epochs = []
        self.Validlogdet_MeanBatch_Epochs = []

        self.latent_dimension = latent_dim


    def loss_function(self, log_p, logdet):

        nll = -(log_p + logdet).mean() ### Average over all samples

        log_p = log_p.detach().cpu().numpy().mean()
        logdet = logdet.detach().cpu().numpy().mean()

        return  nll, log_p, logdet


    def training(self, plot_epochs=True):
        pbar = tqdm(total=self.epochs, desc="Epochs training...")
        best_val_loss = float('inf')
        best_model_weights = None
        for epoch in range(self.epochs):
            ### Training phase

            self.model.train()
            train_total_loss = []
            train_logp_loss = []
            train_logdet_loss = []
            for x_train, y_train in self.train_loader:

                x_train = x_train.to(self.device)
                y_train = y_train.to(self.device)

                self.optimizer.zero_grad()

                ### Forward propagation
                x_pred, log_p, log_det = self.model(x_train, y_train, compute_likelihoods=False)

                loss_train, log_p, log_det = self.loss_function(log_p=log_p, logdet=log_det)

                loss_train.backward()
                self.optimizer.step()

                train_total_loss.append(loss_train.item())
                train_logp_loss.append(log_p)
                train_logdet_loss.append(log_det)

            TrainTotal_MeanBatch = np.mean(train_total_loss)
            Trainlogp_MeanBatch = np.mean(train_logp_loss)
            Trainlogdet_MeanBatch = np.mean(train_logdet_loss)

            self.TrainTotal_MeanBatch_Epochs.append(TrainTotal_MeanBatch)
            self.Trainlog_p_MeanBatch_Epochs.append(Trainlogp_MeanBatch)
            self.Trainlogdet_MeanBatch_Epochs.append(Trainlogdet_MeanBatch)


            ### Validation phase
            self.model.eval()
            valid_total_loss = []
            valid_logp_loss = []
            valid_logdet_loss = []
            with torch.no_grad():
                for x_valid, y_valid in self.valid_loader:

                    x_valid = x_valid.to(self.device)
                    y_valid = y_valid.to(self.device)


                    x_pred_valid, log_p_valid, log_det_valid = self.model(x_valid, y_valid, compute_likelihoods=False)

                    loss_valid, log_p_val, logdet_val = self.loss_function(log_p=log_p_valid, logdet=log_det_valid)

                    valid_total_loss.append(loss_valid.item())
                    valid_logp_loss.append(log_p_val)
                    valid_logdet_loss.append(logdet_val)

                ValidTotal_MeanBatch = np.mean(valid_total_loss)
                Validlogp_MeanBatch = np.mean(valid_logp_loss)
                Validlogdet_MeanBatch = np.mean(valid_logdet_loss)



                self.ValidTotal_MeanBatch_Epochs.append(ValidTotal_MeanBatch)
                self.Validlog_p_MeanBatch_Epochs.append(Validlogp_MeanBatch)
                self.Validlogdet_MeanBatch_Epochs.append(Validlogdet_MeanBatch)

            ## Keep track of the model that results to the minimum validation error
            if ValidTotal_MeanBatch < best_val_loss:
                best_val_loss = ValidTotal_MeanBatch
                best_model_weights = self.model.state_dict()

            if plot_epochs:
                print(f"\nEpoch   Training   Validation logp   logdet    \n"
                      f"{epoch}   {TrainTotal_MeanBatch}  {ValidTotal_MeanBatch} {Validlogp_MeanBatch}   {Validlogdet_MeanBatch} \n"
                      f"====================================================")


            pbar.update()
        pbar.close()
        print("Done training!")

        if best_model_weights:
            self.model.load_state_dict(best_model_weights)

        return self.model


    def plot_losses(self):

        #
        # Plot the losses
        plt.figure()
        plt.loglog(self.TrainTotal_MeanBatch_Epochs,color='blue', label='Total Training')
        plt.loglog(self.ValidTotal_MeanBatch_Epochs, color='orange', label='Total Validation')
        plt.xlabel("# Epochs")
        plt.ylabel("Loss function")
        plt.legend()
        plt.show()

