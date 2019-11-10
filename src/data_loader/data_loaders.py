from src.data_loader.base import BaseDataLoader
from src.dataset import DD40Dataset


class DD40DataLoader(BaseDataLoader):
    """
    Duffy Duck 1940's Data Loader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = DD40Dataset(self.data_dir, train=training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
