import torch
from torch.utils.data import DataLoader
from deepcompression.model.loss import RateDistortionLoss
from tqdm import tqdm


def train_epoch(model, dataloader: DataLoader, device, optimizer, criterion: RateDistortionLoss):
    """
    One training epoch.
    Iterates over dataloader and trains the model.
    Prints loss every 500 batches, and total loss at the end.

    TODO training gives nan from time to time (at least with ERA5 data)
    """

    model = model.to(device)
    model.train()
    all_loss, tmp_loss = {k: 0 for k in ['loss', 'mse_loss', 'bpp_loss']}, {k: 0 for k in ['loss', 'mse_loss', 'bpp_loss']}
    for e, batch in enumerate(tqdm(dataloader)):
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch)
        loss = criterion(out, batch)
        loss['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        for k in loss:
            all_loss[k] += loss[k].item()
            tmp_loss[k] += loss[k].item()

        if e > 0 and e % 500 == 0:
            #print(f"{e}: {tmp_loss['loss']/500:.3f}, MSE loss: {tmp_loss['mse_loss']/500:.3f}, BPP loss: {tmp_loss['bpp_loss']/500:.3f}")
            tmp_loss = {k: 0 for k in ['loss', 'mse_loss', 'bpp_loss']}

    print(f"Training loss: {all_loss['loss']/e:.3f}, MSE loss: {all_loss['mse_loss']/e:.3f}, BPP loss: {all_loss['bpp_loss']/e:.3f}")
    return all_loss


def evaluate(model, dataloader, device, criterion):
    """
    Evaluation over a dataloader, returns loss computed over the dataloader.
    """

    model = model.to(device)
    all_loss = {k: 0 for k in ['loss', 'mse_loss', 'bpp_loss']}
    for e, batch in enumerate(dataloader):
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch)
        all_loss['loss'] += loss['loss'].item()
        all_loss['mse_loss'] += loss['mse_loss'].item()
        all_loss['bpp_loss'] += loss['bpp_loss'].item()

    print(f"Evaluation loss: {all_loss['loss']/e:.3f}, MSE loss: {all_loss['mse_loss']/e:.3f}, BPP loss: {all_loss['bpp_loss']/e:.3f}")
    return all_loss


if __name__=='__main__':

    from deepcompression.model.model import CompressionModel
    from torch import optim


    # Build dataloader
    input_shape, nb_channels = 64, 2
    nb_observations = 1000
    batch_size = 4
    dataset = torch.rand((nb_observations, nb_channels, input_shape, input_shape))
    dataloader = DataLoader(dataset, batch_size)

    # Build model and training parameters
    model = CompressionModel(input_shape, nb_channels)
    device = torch.device('mps')
    optimizer = optim.Adam(model.parameters())
    criterion = RateDistortionLoss(1, input_shape)

    # Train over one epoch and evaluate
    epoch_loss = train_epoch(model, dataloader, device, optimizer, criterion)
    eval_loss = evaluate(model, dataloader, device, criterion)