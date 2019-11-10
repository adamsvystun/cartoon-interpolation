from torchvision import datasets, transforms
from base import BaseDataLoader

from src.dataset import DD40Dataset


class DD40DataLoader(BaseDataLoader):
    """
    Duffy Duck 1940's Data Loader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = DD40Dataset(self.data_dir, train=training, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
