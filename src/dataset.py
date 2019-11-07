from os import listdir
from os.path import join, isdir

import numpy as np
from torch.utils import data
from torchvision import transforms


class CartoonDataset(data.Dataset):

    def __init__(self, directory):
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.triplet_list = np.array([(directory + '/' + f) for f in listdir(directory) if isdir(join(directory, f))])
        self.file_len = len(self.triplet_list)

    def __getitem__(self, index):
        # TODO: Implement getitem
        frame0 = None
        frame1 = None
        frame2 = None

        return frame0, frame1, frame2

    def __len__(self):
        return self.file_len
