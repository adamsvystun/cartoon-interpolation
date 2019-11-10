import os

from torchvision import transforms
from PIL import Image
from skimage import io
from torch.utils import data
import torch
import pandas as pd


class DD40Dataset(data.Dataset):

    def __init__(self, directory, dataset_file, train):
        self.train = train
        self.directory = directory
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(size=(256, 256)),
            transforms.ToTensor(),
        ])
        self.dataset_descriptor = pd.read_csv(os.path.join(directory, dataset_file))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {}
        for i in range(5):
            frame_name = 'frame' + str(i)
            frame_path = os.path.join(self.directory, self.dataset_descriptor.iloc[idx][frame_name])
            frame = Image.open(frame_path)
            sample[frame_name] = self.transform(frame)
            
        return sample

    def __len__(self):
        return len(self.dataset_descriptor.index)
