import os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.io


class ImageNet(Dataset):

    ROOT = 'tmp/imagenet/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC'

    def __init__(self, split: str, size: int, limit: int = None, in_memory: bool = False):
        """
        - split: train, test or val
        - size: resize all images into (size, size)
        - limit: limit number of images
        - in_memory: if True, images are loaded into memory when building the object
        """

        self.limit = limit if limit is not None else np.inf

        self.paths = []
        root = f'{ImageNet.ROOT}/{split}'
        for f in os.listdir(root):
            if f!='.DS_Store':
                if os.path.isdir(f'{root}/{f}'):
                    ps = os.listdir(f'{root}/{f}')
                    self.paths.extend([f'{root}/{f}/{p}' for p in ps])
                else:
                    self.paths.append(f'{root}/{f}')

        self.transforms = transforms.Compose([
            transforms.Resize(size=(size, size)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0]==1 else x), # Some with 1 channel
            transforms.Lambda(lambda x: x[:3]) # Some with 4 channels
        ])

        self.in_memory = in_memory
        if in_memory:
            self.images = [self.transforms(torchvision.io.read_image(p) / 256.) for p in self.paths[:self.limit]]

    def __len__(self):
        return min(len(self.paths), self.limit)

    def __getitem__(self, idx):
        if self.in_memory:
            return self.images[idx]
        else:
            img = torchvision.io.read_image(self.paths[idx]) / 256.
            return self.transforms(img)


if __name__=='__main__':
    
    imagenet = ImageNet('train', 512)
    for e, i in enumerate(imagenet):
        if e%100==0:
            print(e)

    imagenet = ImageNet('train', 512, limit=100, in_memory=True)
    for i in imagenet:
        pass