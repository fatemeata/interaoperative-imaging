import io
import torch
import torchvision
import torch.utils.data as data
import glob
import json
import numpy as np
import os
import skimage
from PIL import Image, ImageDraw


class WristDataset(data.Dataset):

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and json files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.initialize_data(root_dir)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inv_data_path = self.data[idx]
        inv_data = json.load(open(inv_data_path))

        # img_name = inv_data["imagePath"]
        img_name = inv_data_path.rsplit('\\', 1)[-1].split(".")[0] + ".png"
        image_path = os.path.join(inv_data_path.rsplit('\\', 1)[0], img_name)
        image = Image.open(image_path)
        image = torchvision.transforms.Resize((512, 512))(image)
        # transform = torchvision.transforms.PILToTensor()
        # image_tensor = transform(image)

        image = np.expand_dims(np.array(image, dtype=float), axis=0)
        image_tensor = torch.Tensor(image)

        points = [d['points'] for d in inv_data["shapes"] if d['label'] == 'Wrist Joint Line'][0]
        points = np.array([item for sublist in points for item in sublist])
        points = np.float32(points)
        points = np.expand_dims(np.asarray(points), axis=0)
        points = points[np.newaxis, :]
        points_tensor = torch.Tensor(points)

        sample = {'image': image_tensor, 'points': points_tensor}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.data)

    def initialize_data(self, dir_p):
        for subdir, dirs, files in os.walk(dir_p):
            for file in files:
                if file.endswith(".json"):
                    self.data.append(os.path.join(subdir, file))

    def length(self):
        self.__len__()

