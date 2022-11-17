import io
import torch
import torchvision
import torch.utils.data as data
import glob
import json
import numpy as np
import os
import skimage
import PIL
from PIL import Image, ImageDraw
import torchvision.transforms as T
import matplotlib as plt
from skimage.color import rgb2gray

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
        self.image_shape = 512
        self.pad_x_1 = None
        self.pad_y_1 = None
        # create the dataset from json files
        self._initialize_data(root_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        :param idx: index of the item which we try to get the sample
        :return: sample: a dictionary consisting of image tensor and wrist joint line points tensor
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inv_data_path = self.data[idx]
        inv_data = json.load(open(inv_data_path))
        image_tensor = self._get_image_tensor(item_path=inv_data_path)
        points_tensor = self._get_points_tensor(data=inv_data)

        sample = {'image': image_tensor, 'points': points_tensor}

        return sample

    def _initialize_data(self, dir_p):
        """
        :param dir_p: data directory path
        :return: none -> update the list of dataset
        """
        # list all the json files in root directory
        for subdir, dirs, files in os.walk(dir_p):
            for file in files:
                if file.endswith(".json"):
                    self.data.append(os.path.join(subdir, file))

    def _get_image_tensor(self, item_path):
        """
        :param item_path: the path of specific data
        :return: image torch tensor
        """
        img_name = item_path.rsplit('\\', 1)[-1].split(".")[0] + ".png"
        image_path = os.path.join(item_path.rsplit('\\', 1)[0], img_name)
        img = Image.open(image_path)
        grayscale_img = PIL.ImageOps.grayscale(img)
        # img = skimage.io.imread(image_path)
        # grayscale_img = skimage.color.rgb2gray(img)
        padded_image = self._padding(grayscale_img, self.image_shape)
        tensor_conversion = torchvision.transforms.ToTensor()
        image_tensor = tensor_conversion(padded_image)
        return image_tensor

    def _get_points_tensor(self, data):
        """
        :param inv_data: json file of a specific data
        :return: points torch tensor
        """

        points = [d['points'] for d in data["shapes"] if d['label'] == 'Wrist Joint Line'][0]
        points = np.array([item for sublist in points for item in sublist])
        points = np.float32(points)

        # shift the coordinates regarding the padding size
        # start point
        points[0] += self.pad_x_1
        points[1] += self.pad_y_1

        # end point
        points[2] += self.pad_x_1
        points[3] += self.pad_y_1

        points = np.expand_dims(np.asarray(points), axis=0)
        points = points[np.newaxis, :]
        points_tensor = torch.Tensor(points)

        return points_tensor

    def _padding(self, image, image_size):
        """
        :param image: the input image
        :param image_size: final desired image size (in our case 512*512)
        :return: image_t: transformed padded image
        """

        f_x = image.size[0]
        f_y = image.size[1]

        # compute padding size
        pad_x_1 = int((image_size - f_x) / 2)
        pad_y_1 = int((image_size - f_y) / 2)

        if f_x % 2 == 0:  # check the image size is even or not
            pad_x_2 = pad_x_1
        else:  # if the image size is odd, right and left padding is not equal
            pad_x_2 = pad_x_1 + 1

        if f_y % 2 == 0:
            pad_y_2 = pad_y_1

        else:
            pad_y_2 = pad_y_1 + 1

        # store the padding to shift the start and end points coordinates
        self.pad_x_1 = pad_x_1
        self.pad_y_1 = pad_y_1

        # padding
        transform = torchvision.transforms.Pad((pad_x_1, pad_y_1, pad_x_2, pad_y_2))
        image_t = transform(image)

        return image_t
