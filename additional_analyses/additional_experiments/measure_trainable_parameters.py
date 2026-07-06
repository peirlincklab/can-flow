import torch

ae = torch.load('../../data_models_saved/models/ae_model.pth')
cnf = torch.load('../../data_models_saved/models/cnf_model.pth')

cvae = torch.load('../../data_models_saved/models/cvae_model_beta_0.01.pth')

ae_params = sum(
    p.numel() for p in ae.parameters() if p.requires_grad
)

cnf_params = sum(
    p.numel() for p in cnf.parameters() if p.requires_grad
)

cvae_params = sum(
    p.numel() for p in cvae.parameters() if p.requires_grad
)


print('CAN-FLOW # parameters:', cnf_params+ae_params)
print('cVAE # parameters:', cvae_params)