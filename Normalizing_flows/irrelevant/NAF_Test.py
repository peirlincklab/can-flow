import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class MADE_MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim, bias=True)


    def forward(self, x, mask, final_layer=False):
        masked_w = self.linear.weight * mask
        out = torch.nn.functional.linear(x, masked_w, self.linear.bias)


        if not final_layer:
            out = F.leaky_relu(out)

        return out


class AutoregressiveFlow(nn.Module):
    def __init__(self, in_dim, out_dim, final_layer=False):
        super().__init__()

        self.net_s = MADE_MLP(in_dim, out_dim).to('cuda')
        self.net_t = MADE_MLP(in_dim, out_dim).to('cuda')
        self.dim = in_dim

        self.final_layer = final_layer

    def forward(self, x, mask):

        s = self.net_s(x, mask, self.final_layer)
        t = self.net_t(x, mask, self.final_layer)
        z = (x - t) * torch.exp(-s)

        log_det = -torch.sum(s, dim=1)

        return z, log_det


    def reverse(self, z, mask):
        log_det = 0

        s = self.net_s(z, mask, self.final_layer)
        t = self.net_t(z, mask, self.final_layer)

        x = z * torch.exp(s) + t
        log_det += s
        
        return x, log_det
    


class NormalizingFlow(nn.Module):
    def __init__(self, in_dim, out_dim, n_flow):
        super().__init__()


        self.mask = self.initialize_W(in_dim)
        self.ar_flows = nn.ModuleList()
        for i in range(n_flow):
            if i == n_flow - 1:
                self.ar_flows.append(AutoregressiveFlow(in_dim=in_dim, out_dim=out_dim, final_layer=True).to('cuda'))
            else:
                self.ar_flows.append(AutoregressiveFlow(in_dim=in_dim, out_dim=out_dim, final_layer=False).to('cuda'))

        mean = torch.zeros(in_dim).to('cuda')
        cov = torch.eye(in_dim).to('cuda')

        self.prior = torch.distributions.MultivariateNormal(mean, cov)

    def iterate(self, data, direction=1):
        log_det = torch.zeros(data.shape[0], device='cuda')
        for af_flow in self.ar_flows[::direction]:
            cur_flow = af_flow.forward if direction == 1 else af_flow.reverse
            data, ld = cur_flow(data, self.mask)
            if direction == 1:
                log_det += ld
            else:
                log_det = None
        return data, log_det

    def forward(self, x):
        z, log_det = self.iterate(x, direction=1)
        prior_logprob = self.prior.log_prob(z)
        return z, prior_logprob, log_det, self.mask

    def backward(self, z):
        return self.iterate(z, direction=-1)[0]

    def sample(self, x_conf):

        num_samples = x_conf.shape[0]

        z = self.prior.sample((num_samples,))

        z[:, 50:] = x_conf

        x, _ = self.iterate(z, direction=-1)
        return x


    def initialize_W(self, in_dim):

        adj_matrix = 0.01 * torch.ones(in_dim, in_dim, device='cuda')

        n_latent = 50
        n_nodes = 3

        for obs in range(n_latent, n_nodes+n_latent):
            adj_matrix[obs, :n_latent] = 0.01

        adj_matrix[50, n_latent:] = 0
        adj_matrix[52, n_latent:] = 0

        adj_matrix[51, 50] = 0.2
        adj_matrix[51, 52] = 0.2

        for i in range(n_nodes+n_latent):
            adj_matrix[i, i] = 0

        return nn.Parameter(adj_matrix, requires_grad=True).to('cuda')





class Trainer:
    def __init__(self, model, optimizer, epochs, train_loader, valid_loader,
                 latent_dim, alpha, rho_dual, l1_w, omega_tol):
        self.device = torch.device('cuda')

        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.TrainTotal_MeanBatch_Epochs = []
        self.Train_w_MeanBatch_Epochs = []

        self.ValidTotal_MeanBatch_Epochs = []
        self.Valid_w_MeanBatch_Epochs = []

        self.latent_dimension = latent_dim

        ### Define hyperparameters
        self.alpha = alpha
        self.rho_dual = rho_dual
        self.l1_weight = l1_w

        self.omega_tol = omega_tol


    def correct_W(self, adj_matrix):
        n_latent = 50
        n_nodes = 3

        adj_matrix[50, n_latent:] = 0
        adj_matrix[52, n_latent:] = 0

        adj_matrix[51, 50] = 1
        adj_matrix[51, 52] = 1

        for i in range(n_nodes):
            adj_matrix[i, i] = 0

        return adj_matrix


    def get_power_trace(self, A):

        alpha = self.alpha

        B = (torch.eye(self.latent_dimension, device=A.device) + alpha * A ** 2)
        M = torch.matrix_power(B, self.latent_dimension)

        return torch.diag(M).sum() - self.latent_dimension


    def loss_function(self, log_p, logdet, W_mat):


        f_score = -log_p - logdet

        h_w = self.get_power_trace(A=W_mat)

        constr_resid = self.alpha * h_w
        acyclic_term = (self.rho_dual / 2) * h_w ** 2

        l_1_penalty = self.l1_weight * W_mat.abs().mean()

        loss = f_score.mean() + constr_resid + acyclic_term + l_1_penalty

        return loss, h_w


    def training(self):
        pbar = tqdm(total=self.epochs, desc="Epochs training...")
        best_val_loss = float('inf')
        best_model_weights = None
        for epoch in range(self.epochs):
            ### Training phase

            self.model.train()
            train_total_loss = []
            train_w_loss = []
            for x_train in self.train_loader:

                x_train = x_train.to(self.device)

                self.optimizer.zero_grad()

                z, log_p, log_det, W_mat = self.model(x_train)

                # W_mat = self.correct_W(W_mat)

                loss_train, h_w = self.loss_function(log_p=log_p, logdet=log_det, W_mat=W_mat)

                loss_train.backward()
                self.optimizer.step()

                train_total_loss.append(loss_train.item())
                train_w_loss.append(h_w.detach().cpu().numpy())

            TrainTotal_MeanBatch = np.mean(train_total_loss)
            Train_w_MeanBatch = np.mean(train_w_loss)

            self.TrainTotal_MeanBatch_Epochs.append(TrainTotal_MeanBatch)
            self.Train_w_MeanBatch_Epochs.append(Train_w_MeanBatch)

            with torch.no_grad():
                if (epoch + 1) % 15 == 0:
                    ### We perform the dual ascent every 15 epochs
                    self.alpha += self.rho_dual * h_w

            ### Validation phase
            self.model.eval()
            valid_total_loss = []
            valid_w_loss = []
            with torch.no_grad():
                for x_valid in self.valid_loader:

                    x_valid = x_valid.to(self.device)

                    z_pred_valid, log_p_valid, log_det_valid, W_mat_valid = self.model(x_valid)

                    loss_valid, h_w_valid = self.loss_function(log_p=log_p_valid, logdet=log_det_valid, W_mat=W_mat_valid)

                    valid_total_loss.append(loss_valid.item())
                    valid_w_loss.append(h_w_valid.detach().cpu().numpy())

                ValidTotal_MeanBatch = np.mean(valid_total_loss)
                Valid_w_MeanBatch = np.mean(valid_w_loss)


                self.ValidTotal_MeanBatch_Epochs.append(ValidTotal_MeanBatch)
                self.Valid_w_MeanBatch_Epochs.append(Valid_w_MeanBatch)

            ## Keep track of the model that results to the minimum validation error
            if ValidTotal_MeanBatch < best_val_loss:
                best_val_loss = ValidTotal_MeanBatch
                best_model_weights = self.model.state_dict()

            print(f"\nEpoch   Training   Validation h_w  log_p  \n"
                  f"{epoch}   {TrainTotal_MeanBatch}  {ValidTotal_MeanBatch} {Valid_w_MeanBatch} {-log_p.mean() - log_det.mean()} \n"
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
        plt.semilogy(self.Train_w_MeanBatch_Epochs, label='h_w')
        plt.title("Training")
        plt.xlabel("# Epochs")
        plt.ylabel("Loss function")
        plt.legend()
        plt.show()

        plt.figure()
        plt.semilogy(self.ValidTotal_MeanBatch_Epochs, label='Total error')
        plt.semilogy(self.Valid_w_MeanBatch_Epochs, label='h_W')
        plt.xlabel("# Epochs")
        plt.ylabel("Loss function")
        plt.legend()
        plt.show()







