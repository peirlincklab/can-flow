import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm




class ActNorm(nn.Module):
    def __init__(self, num_features, init=True):
        """
        ActNorm layer for an MLP-based normalizing flow.

        Args:
            num_features (int): Number of features in the input.
            init (bool): Whether to initialize using the first batch.
        """
        super().__init__()
        self.num_features = num_features
        self.initialized = not init  # Set to True after first batch

        # Scale (s) and bias (b) parameters
        self.s = nn.Parameter(torch.ones(1, num_features))  # Initialized to 1
        self.b = nn.Parameter(torch.zeros(1, num_features))  # Initialized to 0

    def forward(self, x):
        """
        Forward transformation: y = s * x + b
        """
        if not self.initialized:
            # Compute mean and std for the first batch
            with torch.no_grad():
                mean = x.mean(dim=0, keepdim=True)
                std = x.std(dim=0, keepdim=True)

                # Initialize parameters
                self.b.data.copy_(-mean)
                self.s.data.copy_(1 / (std+ 1e-8))
                self.initialized = True  # Mark as initialized

        # Apply the transformation
        y = self.s * x + self.b

        # Compute log determinant for change of variables
        log_det = torch.sum(torch.log(torch.abs(self.s)))

        return y, log_det

    def reverse(self, y):
        """
        Inverse transformation: x = (y - b) / s
        """
        x = (y - self.b) / self.s

        return x



class InvertibleLinearTransform(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # Initialize a random invertible matrix using LU decomposition
        W = torch.linalg.qr(torch.randn(dim, dim))[0]  # Initialize as an orthogonal matrix
        P, L, U = torch.linalg.lu(W)

        self.P = nn.Parameter(P, requires_grad=False).to('cuda')  # Fixed permutation matrix
        self.L = nn.Parameter(L).to('cuda') # Lower triangular matrix
        self.U = nn.Parameter(U).to('cuda')  # Upper triangular matrix

    def forward(self, x):
        """Applies the invertible linear transformation."""
        W = self.P @ self.L @ self.U  # Reconstruct W
        log_det = self.log_det()

        return x @ W.T, log_det

    def reverse(self, y):
        """Applies the inverse transformation."""
        W_inv = torch.inverse(self.P @ self.L @ self.U)  # Compute W inverse
        return y @ W_inv.T  # Inverse transformation

    def log_det(self):
        """Computes log determinant of W for normalizing flow."""
        diag_U = torch.diag(self.U+ 1e-10)
        return torch.sum(torch.log(torch.abs(diag_U)))  # Log determinant of upper triangular matrix


class AffineCoupling(nn.Module):
    def __init__(self, in_dim, out_dim, h_x_conf_dim):
        super().__init__()

        self.coupling_s = nn.Sequential(
            nn.Linear(int(in_dim/2)+h_x_conf_dim, out_dim),
            nn.LeakyReLU(),
            nn.Linear(out_dim, out_dim),
            nn.LeakyReLU(),
            nn.Linear(out_dim, int(in_dim/2)),
        ).to('cuda')

        self.coupling_t = nn.Sequential(
            nn.Linear(int(in_dim/2)+h_x_conf_dim, out_dim),
            nn.LeakyReLU(),
            nn.Linear(out_dim, out_dim),
            nn.LeakyReLU(),
            nn.Linear(out_dim, int(in_dim/2))
        ).to('cuda')

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

        det_jac = torch.exp(torch.sum(s, dim=1))

        logdet = torch.log(torch.abs(det_jac))

        if not compute_likelihoods:
            return z_out, logdet.mean()
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
    def __init__(self, in_dim, out_dim, h_x_conf_dim):
        super().__init__()

        self.actnorm = ActNorm(num_features=in_dim)
        self.lin_transform = InvertibleLinearTransform(dim=in_dim)
        self.coupling_layer = AffineCoupling(in_dim=in_dim, out_dim=out_dim, h_x_conf_dim=h_x_conf_dim)

    def forward(self, x, h_x_conf):

        out, logdet_actnorm = self.actnorm(x)
        out, logdet_w = self.lin_transform(out)
        out, logdet_coupl = self.coupling_layer(out, h_x_conf)

        logdet = logdet_actnorm + logdet_w + logdet_coupl

        return out, logdet

    def reverse(self, x, h_x_conf):

        inp = self.coupling_layer.reverse(x, h_x_conf)
        inp = self.lin_transform.reverse(inp)
        inp = self.actnorm.reverse(inp)

        return inp


class FlowHeart(nn.Module):
    def __init__(self, in_dim, out_dim, n_flow, in_dim_conf, out_dim_conf):
        super().__init__()

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Block(in_dim=in_dim, out_dim=out_dim, h_x_conf_dim=out_dim_conf).to('cuda'))

        # self.mean_prior = torch.zeros(in_dim).to('cuda')
        # self.cov_prior = torch.eye(in_dim).to('cuda')

        self.conf_embedding = nn.Sequential(
            nn.Linear(in_dim_conf, out_dim_conf),
            nn.LeakyReLU(),
            nn.Linear(out_dim_conf, out_dim_conf)
        ).to('cuda')

        self.mean_prior = nn.Sequential(
            nn.Linear(out_dim_conf, int(in_dim/2)),
            nn.BatchNorm1d(int(in_dim/2)),
            nn.LeakyReLU(),
            nn.Linear(int(in_dim/2), in_dim)
        ).to('cuda')

        self.var_prior = nn.Sequential(
            nn.Linear(out_dim_conf, int(in_dim / 2)),
            nn.BatchNorm1d(int(in_dim / 2)),
            nn.LeakyReLU(),
            nn.Linear(int(in_dim / 2), in_dim),
            nn.Softplus()
        ).to('cuda')


    def forward(self, x, x_conf, compute_likelihoods):

        h_x_conf = self.conf_embedding(x_conf)

        mean = self.mean_prior(h_x_conf)
        var = self.var_prior(h_x_conf)
        cov = torch.diag_embed(var)

        prior = torch.distributions.MultivariateNormal(mean, cov)

        logdet = 0

        for flow in self.flows:
            x, det = flow(x, h_x_conf)
            logdet += det

        if not compute_likelihoods:
            log_p = prior.log_prob(x)
            log_p = torch.mean(log_p)

            return x, log_p, logdet
        else:
            log_p = prior.log_prob(x)

            p_total = log_p + logdet

            return p_total


    def reverse(self, x_conf, animate=False):

        h_x_conf = self.conf_embedding(x_conf)

        mean = self.mean_prior(h_x_conf)
        var = self.var_prior(h_x_conf)
        cov = torch.diag_embed(var)

        prior = torch.distributions.MultivariateNormal(mean, cov)

        outs = []
        ### Sample from the prior
        x = prior.sample((1,))
        x = x.reshape(x_conf.shape[0], mean.shape[1])

        outs.append(x)
        for flow in self.flows[::-1]:
            x = flow.reverse(x, h_x_conf)
            outs.append(x)

        if not animate:
            return x
        else:
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

        nll = -log_p - logdet

        log_p = -log_p.detach().cpu().numpy()
        logdet = -logdet.detach().cpu().numpy()

        return  nll, log_p, logdet


    def training(self):
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

        ### Plot the losses
        plt.figure()
        plt.semilogy(self.TrainTotal_MeanBatch_Epochs, label='Total error')
        plt.semilogy(self.Trainlog_p_MeanBatch_Epochs, label='Log_p')
        plt.semilogy(self.Trainlogdet_MeanBatch_Epochs, label='Log_det')
        plt.title("Training")
        plt.xlabel("# Epochs")
        plt.ylabel("Loss function")
        plt.legend()
        plt.show()

        plt.figure()
        plt.semilogy(self.ValidTotal_MeanBatch_Epochs, label='Total error')
        plt.semilogy(self.Validlog_p_MeanBatch_Epochs, label='Log_p')
        plt.semilogy(self.Validlogdet_MeanBatch_Epochs, label='Log_det')
        plt.xlabel("# Epochs")
        plt.ylabel("Loss function")
        plt.legend()
        plt.show()
































