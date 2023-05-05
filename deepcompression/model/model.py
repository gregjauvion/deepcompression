from deepcompression.model.gdn import GDN
from deepcompression.model.distribution import Logistic, Custom
from enum import Enum
import torch
from torch import nn


class DistributionType(Enum):
    LOGISTIC = 0
    CUSTOM = 1


class CompressionModel(nn.Module):
    """
    This class is a compression model for 3D data (channel, height, width).
    It implements a model similar to the one described in this paper: https://arxiv.org/pdf/1611.01704.pdf.

    The model is composed of:
    - An encoder with 4 blocks returning (channel_last, height // 2**4, width // 2**4)
        (3, 3) convolutional layers, stride=2
        GDN as activation function
    - Latent tensor of shape (channel_last, height // 2**4, width // 2**4)
        Assumed to follow a logistic distribution with channel_last parameters
        Perturbed with U(-0.5, 0.5) during training, rounded during inference
    - A decoder with 4 blocks
        (3, 3) ConvTranspose layers
        GDN inverse
    """

    def __init__(self, nb_channels, distribution=DistributionType.LOGISTIC):
        """
        :param nb_channels: number of channels in the input data sent to the model
        """

        super().__init__()
        
        self.nb_channels = nb_channels

        # Convolutional layers
        self.nb_kernels = 64
        self.g_a = nn.Sequential(
            nn.Conv2d(nb_channels, self.nb_kernels, kernel_size=3, padding=1, stride=2),
            GDN(self.nb_kernels),
            nn.Conv2d(self.nb_kernels, self.nb_kernels, kernel_size=3, padding=1, stride=2),
            GDN(self.nb_kernels),
            nn.Conv2d(self.nb_kernels, self.nb_kernels, kernel_size=3, padding=1, stride=2),
            GDN(self.nb_kernels),
            nn.Conv2d(self.nb_kernels, self.nb_kernels, kernel_size=3, padding=1, stride=2),
            GDN(self.nb_kernels),
        )

        # Latent tensor y has shape (batch, nb_kernels, height, width)
        # y[*, i, *, *] is assumed to follow one of 2 distributions
        # - logistic distribution with loc=0 and scale a trainable parameter
        # - the custom distrbution introduced in https://arxiv.org/pdf/1802.01436.pdf ($6.1)
        if distribution==DistributionType.LOGISTIC:
            self.latent_distribution = Logistic(self.nb_kernels)
            self.register_parameter("log_scales", self.latent_distribution.log_scales)
        elif distribution==DistributionType.CUSTOM:
            self.latent_distribution = Custom(self.nb_kernels, [3, 3, 3])
            for k, v in self.latent_distribution.parameters.items():
                self.register_parameter(k, v)

        # Deconvolutional layers
        self.g_s = nn.Sequential(
            nn.ConvTranspose2d(self.nb_kernels, self.nb_kernels, kernel_size=3, stride=2, padding=1, output_padding=1),
            GDN(self.nb_kernels, inverse=True),
            nn.ConvTranspose2d(self.nb_kernels, self.nb_kernels, kernel_size=3, stride=2, padding=1, output_padding=1),
            GDN(self.nb_kernels, inverse=True),
            nn.ConvTranspose2d(self.nb_kernels, self.nb_kernels, kernel_size=3, stride=2, padding=1, output_padding=1),
            GDN(self.nb_kernels, inverse=True),
            nn.ConvTranspose2d(self.nb_kernels, nb_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def encoder(self, x):
        """"
        :param x: shape (batch, nb_channels, height, width). Outputs (batch, nb_kernels, height//16, width//16)
        """
        return self.g_a(x)

    def decoder(self, y):
        """
        :param y: shape (batch, nb_kernels, h, w). Outputs (batch, nb_channels, h * 16, w * 16)
        """
        return self.g_s(y)

    def forward(self, x):
        """"
        :param x: shape (batch, nb_channels, height, width).

        Outputs dict with:
        - x_hat of shape (batch, nb_channels, height, width)
        - y of shape (batch, nb_kernels, height // 16, width // 16)
        - likelihoods of shape (batch)
        """

        # Encoder
        y = self.encoder(x)

        # Perturb latent vectors with U(-0.5, 0.5) and compute likelihoods
        y_ = y + torch.rand(y.shape, device=y.device) - 0.5
        lls = - torch.sum(torch.log2(self.latent_distribution.get_density(y_, device=y.device) + 1e-12), axis=(1, 2, 3))

        # Decoder
        x_hat = self.decoder(y_)

        return {'x_hat': x_hat, 'y': y, 'likelihoods': lls}


if __name__=='__main__':

    input_shape, nb_channels = 64, 3
    model = CompressionModel(nb_channels, distribution=DistributionType.CUSTOM)

    # Print number of parameters for all layers
    for n, p in model.named_parameters():
        print(n, p.shape)

    # Perform some inferences
    x = torch.rand((4, nb_channels, input_shape, input_shape))
    y = model.encoder(x)
    x_hat = model.decoder(y)
    print(x.shape, y.shape, x_hat.shape)

    o = model(x)
    print(o['x_hat'].shape, o['y'].shape, o['likelihoods'].shape)