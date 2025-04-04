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
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.coupling_s = nn.Sequential(
            nn.Linear(int(in_dim/2), out_dim),
            nn.LeakyReLU(),
            nn.Linear(out_dim, out_dim),
            nn.LeakyReLU(),
            nn.Linear(out_dim, int(in_dim/2)),
        ).to('cuda')

        self.coupling_t = nn.Sequential(
            nn.Linear(int(in_dim/2), out_dim),
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

    def forward(self, x):

        x_a, x_b = torch.chunk(x, 2, dim=1)

        s = self.coupling_s(x_a)
        t = self.coupling_t(x_a)

        z_a = x_a
        z_b = x_b * torch.exp(s) + t

        z_out = torch.cat((z_a, z_b), dim=1)

        det_jac = torch.exp(torch.sum(s, dim=1))

        logdet = torch.log(torch.abs(det_jac))

        return z_out, logdet.mean()


    def reverse(self, z):
        z_a, z_b = torch.chunk(z, 2, dim=1)

        s = self.coupling_s(z_a)
        t = self.coupling_t(z_a)

        x_a = z_a
        x_b = (z_b - t) * torch.exp(-s)

        x_out = torch.cat((x_a, x_b), dim=1)

        return x_out


class Block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.actnorm = ActNorm(num_features=in_dim)
        self.lin_transform = InvertibleLinearTransform(dim=in_dim)
        self.coupling_layer = AffineCoupling(in_dim=in_dim, out_dim=out_dim)

    def forward(self, x):

        out, logdet_actnorm = self.actnorm(x)
        out, logdet_w = self.lin_transform(out)
        out, logdet_coupl = self.coupling_layer(out)

        logdet = logdet_actnorm + logdet_w + logdet_coupl

        return out, logdet

    def reverse(self, x):

        inp = self.coupling_layer.reverse(x)
        inp = self.lin_transform.reverse(inp)
        inp = self.actnorm.reverse(inp)

        return inp


class Flow(nn.Module):
    def __init__(self, in_dim, out_dim, n_flows, split=True):
        """

        :param in_dim:
        :param out_dim:
        :param n_flows: How many flows before the split
        """
        super().__init__()

        self.split = split

        self.flow_blocks = nn.ModuleList()
        for i in range(n_flows):
            self.flow_blocks.append(Block(in_dim, out_dim)).to('cuda')

        # self.split_prior_mean = nn.Sequential(
        #     nn.Linear(int(in_dim / 2), out_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(out_dim, out_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(out_dim, int(in_dim / 2))
        # ).to('cuda')
        #
        # self.split_prior_logvar = self.split_prior_mean = nn.Sequential(
        #     nn.Linear(int(in_dim / 2), out_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(out_dim, out_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(out_dim, int(in_dim / 2))
        # ).to('cuda')

        # if not split:
        if split:
            self.mean_prior = torch.zeros(int(in_dim/2)).to('cuda')
            self.cov_prior = torch.eye(int(in_dim/2)).to('cuda')
            self.prior = torch.distributions.MultivariateNormal(self.mean_prior, self.cov_prior)
        else:
            self.mean_prior = torch.zeros(in_dim).to('cuda')
            self.cov_prior = torch.eye(in_dim).to('cuda')
            self.prior = torch.distributions.MultivariateNormal(self.mean_prior, self.cov_prior)

    def forward(self, x):

        logdet = 0

        for flow in self.flow_blocks:
            x, det = flow(x)
            logdet += det

        if self.split:

            self.out, z_new = torch.chunk(x, 2, dim=1)

            # mean = self.split_prior_mean(z_new)
            # logvar = self.split_prior_logvar(z_new)
            #
            # var = torch.exp(logvar)
            # cov = torch.diag_embed(var)
            #
            # self.split_prior = torch.distributions.MultivariateNormal(mean, cov)
            # log_p_prior = self.split_prior.log_prob(self.out)
            log_p_prior = self.prior.log_prob(self.out)

            return z_new, log_p_prior.mean(), logdet, self.out

        else:
            log_p = self.prior.log_prob(x)
            log_p = torch.mean(log_p)


            return x, log_p, logdet, x

    def reverse(self, x):

        if self.split:
            num_samples = x.shape[0]

            # x_in = self.split_prior.sample((num_samples, ))
            x_in = self.prior.sample((num_samples,))
            x_in = x_in.reshape(num_samples, -1)
            x = torch.cat((x, x_in), dim=1)
        else:
            x = self.prior.sample((x, )) ### TODO: Check this

        for flow in self.flow_blocks[::-1]:
            x = flow.reverse(x)

        return x








class FlowHeart(nn.Module):
    def __init__(self, in_dim, out_dim, n_flow, n_block):
        super().__init__()

        self.flows = nn.ModuleList()
        for i in range(n_flow): ### This is the number of more general flows (coupl + split)
            self.flows.append(Flow(in_dim=in_dim, out_dim=out_dim, n_flows=n_block, split=True))
            out_dim *= 2 ### Make the networks deeper
            in_dim = int(in_dim / 2)
        self.flows.append(Flow(in_dim=in_dim, out_dim=out_dim, n_flows=n_block, split=False))


    def forward(self, x):
        log_p_sum = 0
        logdet_sum = 0
        z_outs = []
        for flow in self.flows:
            x, log_p, logdet, z_out = flow(x)
            z_outs.append(z_out)

            logdet_sum += logdet
            log_p_sum += log_p

        return z_outs, log_p_sum, logdet_sum


    def reverse(self, num_samples):

        for i, flow in enumerate(self.flows[::-1]):

            if i == 0:
                x = num_samples
                input = flow.reverse(x)

            else:
                input = flow.reverse(input)

        return input



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
            for x_train in self.train_loader:

                x_train = x_train.to(self.device)

                self.optimizer.zero_grad()

                ### Forward propagation
                x_pred, log_p, log_det = self.model(x_train)

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
                for x_valid in self.valid_loader:

                    x_valid = x_valid.to(self.device)


                    x_pred_valid, log_p_valid, log_det_valid = self.model(x_valid)

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
































