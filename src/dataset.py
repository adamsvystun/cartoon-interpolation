import os

from torchvision import transforms
from skimage import io
import torch
from torch.utils import data
import pandas as pd


class DD40Dataset(data.Dataset):

    def __init__(self, directory, dataset_file, train):
        self.train = train
        self.directory = directory
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale()
        ])
        self.dataset_descriptor = pd.read_csv(os.path.join(directory, dataset_file))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {}
        for i in range(5):
            frame_name = 'frame' + str(i)
            frame_path = os.path.join(self.directory, self.dataset_descriptor.iloc[idx][frame_name])
            frame = io.imread(frame_path)
            sample[frame_name] = frame

        sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.dataset_descriptor.index)
