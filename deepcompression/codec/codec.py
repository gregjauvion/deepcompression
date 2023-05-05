import torch
from torch.utils.data import DataLoader
import constriction
import numpy as np
from tqdm import tqdm


class Codec:
    def __init__(self, model):

        model.eval()
        self.model = model

        # For each latent dimension, we determine a range of possible symbols
        # (to reduce the number of symbols as the distributions are very concentrated)
        # and we build the categorical entropy model
        xs = torch.Tensor(np.arange(-128, 127)).reshape(1, -1).tile(model.latent_distribution.dimension, 1).transpose(1, 0)
        with torch.no_grad():
            ps = (model.latent_distribution.get_cumulative(xs + 0.5) - model.latent_distribution.get_cumulative(xs - 0.5)).squeeze().numpy()

        #ps[np.isnan(ps)] = 0.
        self.ranges, self.e_models = [], []
        for d in range(ps.shape[1]):
            p = ps[:, d]
            w = np.where(p>1e-4)[0]
            r = (w[0].item(), w[-1].item() + 1)
            self.ranges.append(r)

            probas = p[r[0]:r[1]] / p[r[0]:r[1]].sum()
            self.e_models.append(constriction.stream.model.Categorical(probas.astype(np.float64)))

    def compress(self, dataset, device, batch_size=512):

        # Build latent predictions
        self.model = self.model.to(device)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        latents = []
        for b in tqdm(dataloader):
            b = b.to(device)
            out = self.model.encoder(b)
            latents.append(out.cpu().detach().round().numpy())

        latents = np.concatenate(latents).astype(np.int32)

        # Code latents dimension per dimension for optimization purpose
        coder = constriction.stream.stack.AnsCoder()
        for dim in range(latents.shape[1]):
            to_encode = (latents[:, dim] + 128).clip(min=self.ranges[dim][0], max=self.ranges[dim][1] - 1) - self.ranges[dim][0]
            coder.encode_reverse(to_encode.flatten(), self.e_models[dim])

        return coder.get_compressed(), latents.shape

    def decompress(self, stream, shape, device, batch_size=512):

        # Decompress latents dimension per dimension (in reverse)
        decoder = constriction.stream.stack.AnsCoder(stream)
        latents = np.zeros(shape)
        for dim in range(shape[1]-1, -1, -1):
            dec = decoder.decode(self.e_models[dim], shape[0] * shape[2] * shape[3]).reshape(shape[0], shape[2], shape[3])
            latents[:, dim, :, :] = dec  - 128 + self.ranges[dim][0]

        # Reconstruct data
        data = []
        self.model = self.model.to(device)
        dataloader = DataLoader(torch.Tensor(latents), batch_size=batch_size, shuffle=False)
        for b in tqdm(dataloader):
            b = b.to(device).type(torch.float32)
            out = self.model.decoder(b)
            data.append(out.cpu().detach())

        return torch.concat(data)


if __name__=='__main__':

    from deepcompression.model.model import CompressionModel

    input_shape, nb_channels = 64, 3
    model = CompressionModel(nb_channels)
    codec = Codec(model)

    # Compress and decompress some date
    data = torch.rand((1000, nb_channels, input_shape, input_shape))
    stream, shape = codec.compress(data, torch.device('mps'))
    decomp = codec.decompress(stream, shape, torch.device('mps'))