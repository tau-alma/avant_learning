import torch
import torch.nn.functional as F
from torch import nn
from ogm_encoder.blocks import ConvLayer, ResBlock, ConvLayerTranspose

class BaseVAE(nn.Module):
    def __init__(self):
        super(BaseVAE, self).__init__()

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class VAE(BaseVAE):
    def __init__(self, input_shape, latent_shape):
        super(VAE, self).__init__()
        self.latent_shape = latent_shape

        # Encoder
        self.encoder = nn.Sequential(
            ConvLayer(1, 32, stride=2),
            ConvLayer(32, 64, stride=2),
            nn.MaxPool2d(2),
            ConvLayer(64, 96, stride=2),
            nn.MaxPool2d(2),
            ConvLayer(96, 128, stride=2),
        )

        dummy = torch.zeros((1, 1, input_shape, input_shape))
        dummy_output = self.encoder(dummy)        
        final_shape = dummy_output[0].shape
        print(final_shape)
        flattened = F.adaptive_max_pool2d(dummy_output, 1)
        # flattened = dummy_output.flatten(start_dim=1)
        flattened_shape = flattened.shape[1]
        print(flattened_shape)

        self.encoder.add_module("global_pool", nn.AdaptiveMaxPool2d(1))
        self.encoder.add_module("flatten", nn.Flatten())
        self.encoder.add_module("dense", nn.Linear(flattened_shape, 2*latent_shape))

        initial_shape = (16, 8, 8)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_shape, 16*8*8),
            nn.Unflatten(1, initial_shape),

            ConvLayerTranspose(16, 32, kernel_size=3, stride=2, padding=1),
            ConvLayerTranspose(32, 64, kernel_size=3, stride=2, padding=1),
            ConvLayerTranspose(64, 32, kernel_size=5, stride=4, padding=1),
            ConvLayerTranspose(32, 16, kernel_size=5, stride=4, padding=1),

            # Terminal head:
            nn.Conv2d(16, 3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.Softmax(dim=1)
        )

        dummy = torch.zeros((1, latent_shape))
        dummy_output = self.decoder(dummy)  
        print(dummy_output.shape)

    def encode(self, x):
        mu_logvar = self.encoder(x)
        mu_logvar = mu_logvar.view(-1, 2, self.latent_shape)  # Splitting into mu and logvar
        return mu_logvar[:, 0, :], mu_logvar[:, 1, :]

    def decode(self, z):
        return self.decoder(z)

if __name__ == "__main__":
    vae = VAE(216, 256)

    from dataset import OccupancyGridDataset
    dataset = OccupancyGridDataset("/home/aleksi/ogm_images/")
    loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

    for data, _ in loader:
        recon, mu, logvar = vae(data)
        print(data.shape, recon.shape)
