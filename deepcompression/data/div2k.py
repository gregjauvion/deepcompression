import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.io


class Div2k(Dataset):

    ROOT = 'tmp/DIV2K/DIV2K_valid_HR'

    def __init__(self, size: int):
        """
        - size: resize all images into (size, size)
        """

        self.transforms = transforms.Compose([
            transforms.Resize(size=(size, size)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0]==1 else x), # Some with 1 channel
            transforms.Lambda(lambda x: x[:3]) # Some with 4 channels
        ])
        self.paths = [p for p in os.listdir(Div2k.ROOT) if p!='.DS_Store']
        self.images = [self.transforms(torchvision.io.read_image(f'{Div2k.ROOT}/{p}') / 256.) for p in self.paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.images[idx]


if __name__=='__main__':

    import matplotlib.pyplot as plt

    div2k = Div2k(2048)

    plt.imshow(div2k[0].permute((1, 2, 0)))
    plt.show()