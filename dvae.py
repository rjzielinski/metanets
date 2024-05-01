import matplotlib.pyplot as plt
import nibabel as nib
import nilearn
import numpy as np
import os
import pandas as pd
import seaborn as sns
import torch
import torch.distributions
import torch.nn.functional as F
import torch.utils
import torchvision

from dataclasses import dataclass
from datetime import datetime
from nilearn import datasets, plotting
from nilearn.maskers import NiftiMasker, NiftiMapsMasker
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import v2
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

data_dir = "data/"
atlas = nilearn.datasets.fetch_atlas_msdl()
atlas_filename = atlas["maps"]
labels = atlas["labels"]

data = nilearn.datasets.fetch_development_fmri(data_dir=data_dir)

n_subjects = 30

df_list = list()
masked_list = list()
affine_list = list()
for subj_idx in range(n_subjects):
    df_temp = nilearn.image.get_data(data.func[subj_idx])
    df_affine = nib.load(data.func[subj_idx]).affine

    df_list.append(df_temp)
    affine_list.append(df_affine)

    masker = NiftiMasker(mask_strategy="whole-brain-template")
    masker.fit(data.func[subj_idx])
    fmri_masked = masker.transform(data.func[subj_idx])
    masked_list.append(fmri_masked)

dfs = np.array(df_list)
masked = np.array(masked_list)

df_train = torch.tensor(dfs[0:20])
df_test = torch.tensor(dfs[21:30])

train_loader = torch.utils.data.DataLoader(
    df_train,
    batch_size=10,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    df_test,
    batch_size=batch_size
)

@dataclass
class DVAEOutput:
    z_dist: torch.distributions.Distribution
    z_sample: torch.Tensor
    x_hat: torch.Tensor

    loss: torch.Tensor
    loss_recon: torch.Tensor
    loss_kl: torch.Tensor


class DVAE(nn.Module):
    def __init__(self, input_dim, input_size, hidden_dim, latent_dim):
        super(DVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 80),
            nn.Tanh(),
            nn.Linear(80, 2 * latent_dim),
        )

        self.softplus = nn.Softplus()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 80),
            nn.Tanh(),
            nn.Linear(80, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, input_size),
            nn.Sigmoid(),
            nn.Unflatten(1, input_dim),
        )

    def encode(self, x, eps: float = 1e-8):
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=-1)
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)

        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)
    
    def reparametrize(self, dist):
        return dist.rsample()
    
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, compute_loss: bool = True):
        dist = self.encode(x)
        z = self.reparametrize(dist)
        x_hat = self.decode(z)

        if not compute_loss:
            return DVAEOutput(
                z_dist=dist,
                z_sample=z,
                x_hat=x_hat,
                loss=None,
                loss_recon=None,
                loss_kl=None
            )
        
        loss_recon = F.binary_cross_entropy(x_hat, x + 0.5, reduction="none").sum(-1).mean()
        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1),
        )
        loss_kl = torch.distributions.kl.kl_divergence(dist, std_normal).mean()
        loss = loss_recon + loss_kl

        return DVAEOutput(
            z_dist = dist,
            z_sample=z,
            x_hat=x_hat,
            loss=loss,
            loss_recon=loss_recon,
            loss_kl=loss_kl
        )


def train(model, dataloader, optimizer, prev_updates, writer=None):
    model.train()

    for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
        n_upd = prev_updates + batch_idx
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = output.loss
        loss.backward()

        if n_upd % 100 == 0:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)

            print(f'Step {n_upd:,} (N samples: {n_upd*batch_size:,}), Loss: {loss.item():.4f} (Recon: {output.loss_recon.item():.4f}, KL: {output.loss_kl.item():.4f}) Grad: {total_norm:.4f}')
            if writer is not None:
                global_step = n_upd
                writer.add_scalar('Loss/Train', loss.item(), global_step)
                writer.add_scalar('Loss/Train/BCE', output.loss_recon.item(), global_step)
                writer.add_scalar('Loss/Train/KLD', output.loss_kl.item(), global_step)
                writer.add_scalar('GradNorm/Train', total_norm, global_step)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return prev_updates + len(dataloader)

def test(model, dataloader, cur_step, writer=None):
    model.eval()
    test_loss = 0
    test_recon_loss = 0
    test_kl_loss = 0

    with torch.no_grad():
        for data, target in tqdm(dataloader, desc='Testing'):
            data = data.to(device)
            # data = data.view(data.size(0), -1)
            output = model(data, compute_loss=True)

            test_loss += output.loss.item()
            test_recon_loss += output.loss_recon.item()
            rest_kl_loss += output.loss_kl.item()
    
    test_loss /= len(dataloader)
    test_recon_loss /= len(dataloader)
    test_kl_loss /= len(dataloader)
    print(f'====> Test set loss: {test_loss:.4f} (BCE: {test_recon_loss:.4f}, KLD: {test_kl_loss:.4f})')

    if writer is not None:
        writer.add_scalar('Loss/Test', test_loss, global_step=cur_step)
        writer.add_scalar('Loss/Test/BCE', output.loss_recon.item(), global_step=cur_step)
        writer.add_scalar('Loss/Test/KLD', output.loss_kl.item(), global_step=cur_step)

        writer.add_images('Test/Reconstructions', output.x_hat, global_step=cur_step)
        writer.add_images('Test/Originals', data, global_step=cur_step)

        z = torch.randn(16, latent_dim).to(device)
        samples = model.decode(z)
        writer.add_images('Test/Samples', samples, global_step=cur_step)


input_dim = df_train.shape[1:5]
input_size = input_dim[1] * input_dim[2] * input_dim[3] * input_dim[4]
learning_rate = 1e-3
weight_decay = 1e-2
num_epochs = 50
latent_dim = 2
hidden_dim = 512

model = DVAE(input_dim=input_dim, input_size=input_size, hidden_dim=hidden_dim, latent_dim=latent_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
writer = SummaryWriter(f'runs/dvae_{datetime.now().strftime("%Y%m%d-%H%M%S")}')

prev_updates = 0
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    prev_updates = train(model, train_loader, optimizer, prev_updates, writer=writer)
    test(model, test_loader, prev_updates, writer=writer)
