from torch import nn
import torch


class RateDistortionLoss(nn.Module):
    """
    Builds the Rate-Distortion Loss equal to (R + lmbda * D)
    - R: latent vector entropy expressed in bpp (bits per pixel)
    - D: distortion measure, MSELoss for the moment
    """

    def __init__(self, lmbda, patch_size):
        super().__init__()

        self.lmbda = lmbda
        self.patch_size = patch_size
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        """
        - output: dict with x_hat (reconstructed input, of shape (batch_size, *x.shape)), likelihoods (of shape (batch_size, ))
        - target:
        """

        mse_loss = self.mse(output['x_hat'], target) * 100 ** 2
        bpp_loss = torch.mean(output['likelihoods']) / (self.patch_size ** 2)

        return {'mse_loss': mse_loss, 'bpp_loss': bpp_loss, 'loss': bpp_loss + self.lmbda * mse_loss}
