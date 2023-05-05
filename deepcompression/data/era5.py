import pickle
import itertools as itt
import torch
from torch.utils.data import Dataset


NORM = {
    'temperature': torch.Tensor([-50., 50.]),
    'wind_u': torch.Tensor([-25., 25.]),
    'wind_v': torch.Tensor([-25., 25.]),
    'pressure': torch.Tensor([0.5, 1.]),
    'relative_humidity': torch.Tensor([0., 100.])
}


class Era5(Dataset):
    """
    pkl files with ERA5 data over part of the US have been created for year 2021.
    We have monthly files and yearly files per parameter.
    This generator reads the per-parameter yearly files.
    """

    def __init__(self, parameters: list[str], months: list[int], patch_size: int = None, normalized: bool = True):
        """
        - parameters: available parameters are temperature, relative_humidity, pressure, wind_u, wind_v
        - months: available months are 1 to 12
        - patch_size: patch size used to split spatial data. if None, no splits over spatial dimension
        - normalized: when True, data is normalized using min-max normalizer
        """

        self.parameters = parameters
        self.normalized = normalized

        # Read raw ERA5 data for all months
        data = []
        for m in months:
            m_data = pickle.load(open(f'tmp/era5/era5_us_all_2021_{m:02d}.pkl', 'rb'))
            idx = [list(m_data['channels']).index(p) for p in parameters]
            data.append(torch.Tensor(m_data['data'][:, :, :, idx]))

        # Filter bad timestamps where the data is constant equal to 0 (or -273 for temperature)
        all_data = torch.concat(data)
        ts = [t for t in range(all_data.shape[0]) if len(torch.unique(all_data[t, 150:155, 350:355, 0]))>1]
        all_data = all_data[ts]

        # Build spatial splits
        if patch_size is None:
            self.patches = all_data
        else:
            ys, xs = range(0, all_data.shape[1] - patch_size, patch_size), range(0, all_data.shape[2] - patch_size, patch_size)
            self.patches = torch.concat([
                all_data[t:t+1, y:y+patch_size, x:x+patch_size] for t, y, x in itt.product(range(all_data.shape[0]), ys, xs)
            ])

        # Put channels first
        self.patches = torch.permute(self.patches, (0, 3, 1, 2))

    def normalize(self, x):
        """ Normalize data. x can be of shape (channel, *shape) or (batch, channel, *shape). """

        min_max = torch.stack([NORM[p] for p in self.parameters])
        min_, max_ = min_max[:, 0][:, None, None], min_max[:, 1][:, None, None]
        return (x - min_) / (max_ - min_)

    def denormalize(self, x):
        min_max = torch.stack([NORM[p] for p in self.parameters])
        min_, max_ = min_max[:, 0][:, None, None], min_max[:, 1][:, None, None]
        return min_ + x * (max_ - min_)

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, idx):
        p = self.patches[idx]
        return self.normalize(p) if self.normalized else p


class Patcher:
    """
    Object to build patches from (batch, channel, *shape) data, and back
    """

    def __init__(self, data_shape, patch_size):
        self.data_shape = data_shape
        self.patch_size = patch_size

        # Compute start indices of patches
        self.ys = list(range(0, data_shape[0] - self.patch_size, self.patch_size))
        if self.ys[-1] + self.patch_size < data_shape[0]:
            self.ys.append(data_shape[0] - self.patch_size)

        self.xs = list(range(0, data_shape[1] - self.patch_size, self.patch_size))
        if self.xs[-1] + self.patch_size < data_shape[1]:
            self.xs.append(data_shape[1] - self.patch_size)

    def get_patches(self, data):
        """
        - data: shape (batch, channel, *shape)
        """

        return torch.concat([
            torch.stack([
                data[b, :, y:y + self.patch_size, x:x + self.patch_size] for y, x in itt.product(self.ys, self.xs)
            ])
            for b in range(data.shape[0])
        ])

    def get_images(self, patches):
        """
        - patches: shape (batch, channel, *shape)
        """

        nb_patches_per_image = len(self.xs) * len(self.ys)
        nb_images = patches.shape[0] // nb_patches_per_image
        data = torch.zeros((nb_images, patches.shape[1], *self.data_shape))
        for b in range(nb_images):
            for e, (y, x) in enumerate(itt.product(self.ys, self.xs)):
                data[b, :, y:y + self.patch_size, x:x + self.patch_size] = patches[b * nb_patches_per_image + e]

        return data


if __name__=='__main__':

    import matplotlib.pyplot as plt

    ###
    # Era5 object to build datasets
    ###

    dataset = Era5(['temperature', 'wind_u', 'wind_v', 'pressure', 'relative_humidity'], [1, 2], patch_size=64)
    print(dataset[0].shape)

    dataset_2 = Era5(['temperature', 'wind_u', 'wind_v'], [1, 2], patch_size=None)
    denorm = dataset_2.denormalize(dataset_2[0])
    plt.imshow(denorm[0, :, :]) ; plt.colorbar() ; plt.show()


    ###
    # Test Patcher object
    ###

    data_shape, patch_size = (100, 100), 32
    patcher = Patcher(data_shape, patch_size)

    data = torch.rand((10, 3, *data_shape))
    patches = patcher.get_patches(data)
    data_ = patcher.get_images(patches)

    print((data - data_).abs().max())