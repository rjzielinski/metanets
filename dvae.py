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

class fMRIDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, labels_file=None, transform=None, target_transform=None):
        if labels_file is not None:
            self.img_labels = pd.read_csv(labels_file) 
        else:
            self.img_labels = None
        self.img_dir = img_dir
        self.img_files = [name for name in os.listdir(img_dir) if (os.path.isfile(os.path.join(img_dir, name)) and name.endswith('.gz'))]
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = nilearn.image.get_data(img_path)
        if self.img_labels is not None:
            label = self.img_labels.iloc[idx, 1]
            if self.target_transform:
                label = self.target_transform(label)
        else:
            label = np.array([]) 
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image)
        return image, label

development_fmri = fMRIDataset(img_dir="data/development_fmri/development_fmri")
batch_size = 5
test_split = 0.2
shuffle_dataset = True
random_seed = 42

dataset_size = len(development_fmri)
indices = list(range(dataset_size))
split = int(np.floor(test_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]

train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(
    development_fmri,
    batch_size=batch_size,
    sampler=train_sampler
)
test_loader = torch.utils.data.DataLoader(
    development_fmri,
    batch_size=batch_size,
    sampler=test_sampler
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
            nn.Linear(input_size, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 2 * latent_dim),
        )

        self.softplus = nn.Softplus()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_size),
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


input_dim = next(iter(train_loader))[0].shape[1:5]
input_size = input_dim[0] * input_dim[1] * input_dim[2] * input_dim[3]
learning_rate = 1e-3
weight_decay = 1e-2
num_epochs = 50
latent_dim = 2
hidden_dim = 32

model = DVAE(input_dim=input_dim, input_size=input_size, hidden_dim=hidden_dim, latent_dim=latent_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
writer = SummaryWriter(f'runs/dvae_{datetime.now().strftime("%Y%m%d-%H%M%S")}')

prev_updates = 0
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    prev_updates = train(model, train_loader, optimizer, prev_updates, writer=writer)
    test(model, test_loader, prev_updates, writer=writer)
