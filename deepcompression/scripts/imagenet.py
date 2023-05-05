import os
import numpy as np
from deepcompression.data.imagenet import ImageNet
from deepcompression.data.div2k import Div2k
from deepcompression.model.model import CompressionModel, DistributionType
from deepcompression.model.loss import RateDistortionLoss
from deepcompression.model.train import train_epoch, evaluate
from deepcompression.codec.codec import Codec
from deepcompression.benchmark.jpeg import encode, decode
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt


size = 256
in_memory = True
tr_dataset = ImageNet('train', size, limit=2000, in_memory=in_memory)
te_dataset = ImageNet('test', size, limit=1000, in_memory=in_memory)


###
# Train ImageNet model
###

tr_dataloader = DataLoader(tr_dataset, batch_size=16, shuffle=True)
te_dataloader = DataLoader(te_dataset, batch_size=64, shuffle=True)

# Training parameters
device = torch.device('mps')
distribution, lmbda = DistributionType.CUSTOM, 0.1
criterion = RateDistortionLoss(lmbda, size)
nb_epochs = 4
path = f'tmp/models/imagenet_{lmbda}_{distribution}'
os.makedirs(path, exist_ok=True)

# Build and train compression model
model = CompressionModel(3, distribution=distribution)
print(sum([np.prod(p.size()) for p in model.parameters()]))
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(nb_epochs):
    train_epoch(model, tr_dataloader, device, optimizer, criterion)
    evaluate(model, te_dataloader, device, criterion)
    torch.save(model.state_dict(), f'{path}/imagenet_{epoch}.pt')


###
# Load a model and compress some data
###

distribution = DistributionType.LOGISTIC
path = f'tmp/models/imagenet_0.1_{distribution}/imagenet_3.pt'
device = torch.device('mps')
model = CompressionModel(3, distribution=distribution)
model.load_state_dict(torch.load(path))
codec = Codec(model)

# Compress and decompress some ImageNet images
dataset = ImageNet('train', size, limit=2000, in_memory=True)
stream, shape = codec.compress(dataset, device, batch_size=128)
decomp = codec.decompress(stream, shape, device, batch_size=128)

print('Compression ratio:', int((len(dataset) * 3 * size * size * 4) / (len(stream) * 4)))
for idx in range(0, 2000, 100):
    fig = plt.figure(figsize=(12, 8))
    g = fig.add_subplot(121)
    plt.imshow(dataset[idx].permute((1, 2, 0))) ; plt.axis('off')
    g = fig.add_subplot(122)
    plt.imshow(decomp[idx].permute((1, 2, 0))) ; plt.axis('off')
    plt.savefig(f'tmp/figures/image_{idx}') ; plt.close()

# Compress and decompress some DIV2K images
size = 2048
div2k = Div2k(size)
stream, shape = codec.compress(div2k, device, batch_size=4)
decomp = codec.decompress(stream, shape, device, batch_size=4)

print('Compression ratio:', int((len(div2k) * 3 * size * size * 4) / (len(stream) * 4)))
for idx in range(0, 10):
    fig = plt.figure(figsize=(12, 8))
    g = fig.add_subplot(121)
    plt.imshow(div2k[idx].permute((1, 2, 0))) ; plt.axis('off')
    g = fig.add_subplot(122)
    plt.imshow(decomp[idx].permute((1, 2, 0))) ; plt.axis('off')
    plt.savefig(f'tmp/figures/image_div2k_{idx}') ; plt.close()


###
# Evaluate compression models and jpeg encoding on dataset
###

size, batch_size = 256, 128
dataset = ImageNet('test', size, limit=2000, in_memory=True)
metrics = {}

# Compression models
model = CompressionModel(3)
for lmbda in [0.01, 0.1, 1, 10]:
    model.load_state_dict(torch.load(f'tmp/models/imagenet_{lmbda}/imagenet_14.pt'))
    codec = Codec(model)
    stream, shape = codec.compress(dataset, device, batch_size=batch_size)
    decomp = codec.decompress(stream, shape, device, batch_size=batch_size)
    error = np.mean([((decomp[i] - dataset[i])**2).mean().item() * 100**2 for i in range(decomp.shape[0])])
    ratio = (len(dataset) * 3 * size * size) / (len(stream) * 4)
    metrics[f'model_{lmbda}'] = (error, ratio)

# Jpeg encoding
for q in [10, 25, 50, 75, 95]:
    print(q)
    imgs = [dataset[i].permute(1, 2, 0).numpy() for i in range(len(dataset))]
    enc = [encode(img, jpeg_quality=q) for img in imgs]
    dec = [decode(e) for e in enc]
    error = np.mean([((d - i)**2).mean() * 100**2 for d, i in zip(dec, imgs)])
    ratio = len(imgs) * np.product(imgs[0].shape) / sum([len(e) for e in enc])
    metrics[f'jpeg_{q}'] = (error, ratio)

jpeg_metrics = [metrics[f'jpeg_{q}'] for q in [10, 25, 50, 75, 95]]
model_metrics = [metrics[f'model_{l}'] for l in [0.01, 0.1, 1, 10]]
plt.plot([i[0] for i in jpeg_metrics], [i[1] for i in jpeg_metrics], label='jpeg')
plt.plot([i[0] for i in model_metrics], [i[1] for i in model_metrics], label='model')
plt.xlabel('Error') ; plt.ylabel('Ratio') ; plt.legend() ; plt.grid()
plt.xlim(xmin=0) ; plt.ylim(ymin=0)
plt.show()
