# copied from https://github.com/pytorch/examples/tree/master/mnist
# and from https://github.com/AntixK/PyTorch-VAE/blob/8700d245a9735640dda458db4cf40708caf2e77f/tests/test_vae.py#L11
import argparse
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class ConvVAE(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 default_channels: int = 8,
                 kernel_size: int = 4,
                 latent_dim: int = 32):
        super(ConvVAE, self).__init__()

        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=in_channels, out_channels=default_channels, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.enc2 = nn.Conv2d(
            in_channels=default_channels, out_channels=default_channels * 2, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.enc3 = nn.Conv2d(
            in_channels=default_channels * 2, out_channels=default_channels * 4, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.enc4 = nn.Conv2d(
            in_channels=default_channels * 4, out_channels=64, kernel_size=kernel_size,
            stride=2, padding=0
        )
        # fully connected layers for learning representations
        self.fc1 = nn.Linear(64, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 64)
        self.fc3 = nn.Linear(64, 64*4*4)
        self.fc4 = nn.Linear(64*4*4, 16*20*20)
        self.fc5 = nn.Linear(16*20*20, 8*40*40)
        # decoder
        self.dec1 = nn.ConvTranspose2d(
            in_channels=64, out_channels=default_channels * 8, kernel_size=kernel_size,
            stride=1, padding=0
        )
        # stride 3, padding 2: 9 x 9
        # stride 3, padding 1: 11 x 11
        # stride 3, padding 2, output padding1: 10x10
        self.dec2 = nn.ConvTranspose2d(
            in_channels=default_channels * 8, out_channels=default_channels * 4, kernel_size=kernel_size,
            stride=3, padding=2, output_padding=1
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=default_channels * 4, out_channels=default_channels * 2, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=default_channels * 2, out_channels=default_channels, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.dec5 = nn.ConvTranspose2d(
            in_channels=default_channels, out_channels=in_channels, kernel_size=kernel_size,
            stride=2, padding=1
        )

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample

    def forward(self, input):
        # encoding
        x = F.leaky_relu(self.enc1(input))  # Output shape: (batch, 8, 40, 40)
        x = F.leaky_relu(self.enc2(x))  # Output shape: (batch, 16, 20, 20)
        x = F.leaky_relu(self.enc3(x))  # Output shape: (batch, 32, 10, 10)
        x = F.leaky_relu(self.enc4(x))  # Output shape: (batch, 64, 4, 4)
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)  # Output shape: (batch, 64)
        hidden = F.leaky_relu(self.fc1(x))  # Output shape: (Batch, 128)
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        z = F.leaky_relu(self.fc2(z))  # Output shape: (batch, 64)
        # z = z.view(-1, 64, 1, 1)

        # decoding
        x = F.leaky_relu(self.fc3(z))  # Output shape: (batch, 64 * 4 * 4)
        # x = x.view(-1, 64, 4, 4)
        x = F.leaky_relu(self.fc4(x))  # Output shape: (batch, 16 * 20 * 20)
        x = x.view(-1, 16, 20, 20)
        # x = F.leaky_relu(self.fc5(x))  # Output shape: (batch, 4 * 40 * 40)
        # x = x.view(-1, 8, 40, 40)
        # x = F.leaky_relu(self.dec1(z))  # Output shape: (batch, 64, 4, 4)
        # x = F.leaky_relu(self.dec2(x))  # Output shape: (batch, 32, 10, 10)
        # x = F.leaky_relu(self.dec3(x))  # Output shape: (batch, 16, 20, 20)
        x = F.leaky_relu(self.dec4(x))  # Output shape: (batch, 8, 40, 40)
        output = F.sigmoid(2*self.dec5(x))  # (batch, 1, 80, 80)

        forward_result = dict(
            input=input,
            recon_input=torch.ones_like(input) - output,  # just learn what to subtract
            mu=mu,
            log_var=log_var)

        return forward_result

    def loss_function(self,
                      input,
                      recon_input,
                      mu,
                      log_var,
                      kld_weight: float = 1.) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        """

        # recons_loss = F.mse_loss(recon_input, input)
        recons_loss = F.binary_cross_entropy(input=recon_input, target=input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(),
                                               dim=1),
                              dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}


def vae_load(omniglot_dataset):

    # Train the VAE
    lr = 1e-1
    gamma = 1 
    vae = ConvVAE(in_channels=1)

    save_path = 'vae.pth'
    vae = vae_train_and_save(vae=vae,
                             omniglot_dataset=omniglot_dataset,
                             vae_path=save_path,
                             epochs=1000000,
                             lr=lr,
                             gamma=gamma)
    vae.load_state_dict(torch.load(save_path, map_location='cpu'))
    return vae


def vae_train_and_save(vae,
                       omniglot_dataset,
                       vae_path: str,
                       batch_size: int = 128,
                       epochs: int = 100,
                       lr: float = 1e-2,
                       gamma: float = 0.9,
                       seed: int = 1,
                       log_interval: int = 1):

    omniglot_dataloader = torch.utils.data.DataLoader(dataset=omniglot_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True)
    torch.manual_seed(seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'Device: {device}')
    vae.to(device)

    optimizer = optim.Adadelta(vae.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(epochs + 1):
        vae_train_step(vae=vae, device=device, train_loader=omniglot_dataloader,
                       optimizer=optimizer, epoch=epoch, log_interval=log_interval,
                       vae_path=vae_path)
        scheduler.step()
    return vae


def vae_train_step(vae, device, train_loader, optimizer, epoch, log_interval,
                   vae_path):
    vae.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        forward_result = vae(data)

        # save a few images
        if epoch % 1 == 0 and batch_idx == 0:
            dir_path = vae_path[:-4]
            os.makedirs(dir_path, exist_ok=True)
            for i in range(3):
                fig, axes = plt.subplots(nrows=1, ncols=2)
                axes[0].imshow(forward_result['input'][i, 0, :, :].detach().numpy(), cmap='gray')
                axes[1].imshow(forward_result['recon_input'][i, 0, :, :].detach().numpy(), cmap='gray')
                plt.savefig(os.path.join(f'{dir_path}/epoch={epoch}_image={i}.png'),
                            bbox_inches='tight',
                            dpi=300)

        loss_result = vae.loss_function(
            input=forward_result['input'],
            recon_input=forward_result['recon_input'],
            mu=forward_result['mu'],
            log_var=forward_result['log_var'])
        loss_result['loss'].backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss_result['loss'].item()))

    torch.save(vae.state_dict(), vae_path)

def vae_test_step(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            forward_result = model(data)
            loss_result = model.loss_function(
                input=forward_result['input'],
                recon_input=forward_result['recon_input'],
                mu=forward_result['mu'],
                log_var=forward_result['log_var'])
            test_loss += loss_result['loss']

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

    return test_loss


if __name__ == '__main__':
    vae_load()