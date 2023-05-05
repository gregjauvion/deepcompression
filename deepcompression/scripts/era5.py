import os
import numpy as np
from deepcompression.data.era5 import Era5, Patcher
from deepcompression.model.model import CompressionModel
from deepcompression.model.loss import RateDistortionLoss
from deepcompression.model.train import train_epoch, evaluate
from deepcompression.codec.codec import Codec
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt


parameters = ['temperature', 'wind_u', 'wind_v']
patch_size = 128
tr_months, te_months = [1, 2, 3], [4]
tr_dataset = Era5(parameters, tr_months, patch_size=patch_size)
te_dataset = Era5(parameters, te_months, patch_size=patch_size)


###
# Train ERA5 model
###

tr_dataloader = DataLoader(tr_dataset, batch_size=16, shuffle=True)
te_dataloader = DataLoader(te_dataset, batch_size=128, shuffle=False)

# Training parameters
path = 'tmp/models/era5'
os.makedirs(path, exist_ok=True)
device = torch.device('mps')
criterion = RateDistortionLoss(10, patch_size)
nb_epochs = 15

# Build and train compression model
model = CompressionModel(patch_size, len(parameters))
#model.load_state_dict(torch.load('tmp/models/era5/era5_14.pt'))
print(sum([np.prod(p.size()) for p in model.parameters()]))
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(nb_epochs):
    a, b = train_epoch(model, tr_dataloader, device, optimizer, criterion)
    evaluate(model, te_dataloader, device, criterion)
    torch.save(model.state_dict(), f'{path}/era5_{epoch}.pt')



###
# Load a model and compress some data
###

path = 'tmp/models/era5/era5_14.pt'
model = CompressionModel(patch_size, len(parameters))
model.load_state_dict(torch.load(path))
codec = Codec(model)

# Compress and decompress some data
device = torch.device('mps')
stream, shape = codec.compress(te_dataset, device)
decomp = codec.decompress(stream, shape, device)

print('Compression ratio:', int((len(te_dataset) * 3 * 128 * 128 * 4) / (len(stream) * 4)))

# Plot reconstruction on the whole grid
dataset = Era5(parameters, [1], patch_size=None)
patcher = Patcher(tuple(dataset[0].shape[1:]), patch_size)

data = dataset[:1]
data_comp, shape = codec.compress(patcher.get_patches(data), device)
data_rec = patcher.get_images(codec.decompress(data_comp, shape, device))

data_denorm, data_rec_denorm = dataset.denormalize(data), dataset.denormalize(data_rec)

# Plot with all parameters
fig = plt.figure(figsize=(16, 12))
for c in range(3):
    vmin, vmax = data_denorm[0, c, :, :].min(), data_denorm[0, c, :, :].max()
    g = fig.add_subplot(3, 3, 1 + 3 * c) ; plt.axis('off')
    plt.imshow(data_denorm[0, c, :, :], vmin=vmin, vmax=vmax) ; plt.colorbar()
    plt.title(f'{parameters[c]} (original)')

    g = fig.add_subplot(3, 3, 2 + 3 * c) ; plt.axis('off')
    plt.imshow(data_rec_denorm[0, c, :, :], vmin=vmin, vmax=vmax) ; plt.colorbar()
    plt.title(f'{parameters[c]} (reconstructed)')

    diff = data_rec_denorm[0, c, :, :] - data_denorm[0, c, :, :]
    g = fig.add_subplot(3, 3, 3 + 3 * c) ; plt.axis('off')
    plt.imshow(diff.abs() / 4, vmax=3, cmap='Greys') ; plt.colorbar()
    plt.title(f'{parameters[c]} (difference)')

plt.savefig('tmp/figures/era5') ; plt.close()

# Temperature plot
c = 0
fontsize = 15
vmin, vmax = data_denorm[0, c, :, :].min(), data_denorm[0, c, :, :].max()

plt.axis('off')
plt.imshow(data_denorm[0, c, :, :], vmin=vmin, vmax=vmax) ; plt.colorbar()
plt.title(f'Original data', fontsize=fontsize)
plt.savefig(f'tmp/figures/era5_original_{parameters[c]}') ; plt.close()

plt.axis('off')
plt.imshow(data_rec_denorm[0, c, :, :], vmin=vmin, vmax=vmax) ; plt.colorbar()
plt.title(f'Reconstructed data', fontsize=fontsize)
plt.savefig(f'tmp/figures/era5_reconstructed_{parameters[c]}') ; plt.close()

plt.axis('off')
diff = data_rec_denorm[0, c, :, :] - data_denorm[0, c, :, :]
plt.imshow(diff.abs() / 4, vmax=3, cmap='Greys') ; plt.colorbar()
plt.title(f'Absolute difference', fontsize=fontsize)
plt.savefig(f'tmp/figures/era5_diff_{parameters[c]}') ; plt.close()
