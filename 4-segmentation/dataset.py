import torch
import torchvision
import torch.utils.data as data
import json
import numpy as np
import os
import PIL
from PIL import Image
import re
import enum
from skimage.draw import polygon
from utils.heatmap_creation import Heatmap

class Dataset(data.Dataset):

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
        self._pad_x1 = None
        self._pad_y1 = None
        # create the dataset from json files
        self._initialize_data(root_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        :param idx: index of the item which we try to get the sample
        :return: sample: a dictionary consisting of image tensor, points tensor, patient_id and index
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inv_data_path = self.data[idx]

        #  Data path is different in remote server and local machine. line 46 for remote machine, 47 for local machine
        #  use the folder name as patient id

        # patient_id = re.findall(r'\d+', inv_data_path.split("/")[1])[0]
        patient_id = re.findall(r'\d+', inv_data_path.split("\\")[1])[0]
        with open(inv_data_path) as f:
            inv_data = json.load(f)
        self.image_tensor = self._get_image_tensor(item_path=inv_data_path)
        points_tensor = self._get_points_tensor(data=inv_data)  # it's not torch tensor, np.array
        # mask = self.create_mask(points_tensor, self.image_shape)
        mask = None
        sample = self.create_sample(self.image_tensor, points_tensor, mask, patient_id, idx)

        # Data Augmentation
        if self.transform:
            sample = self.transform(sample)

        self.sample = sample
        return self.sample

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
        #  Data path is different in remote server and local machine.
        #  for remote server use the first two lines 82-83
        #  for local server use the other 84-85

        # img_name = item_path.rsplit("/", 1)[-1].split(".")[0] + ".png"
        # image_path = os.path.join(item_path.rsplit("/", 1)[0], img_name)
        img_name = item_path.rsplit("\\", 1)[-1].split(".")[0] + ".png"
        image_path = os.path.join(item_path.rsplit("\\", 1)[0], img_name)
        print("image path: ", image_path)
        img = Image.open(image_path)
        grayscale_img = PIL.ImageOps.grayscale(img)
        padded_image = self._padding(grayscale_img, self.image_shape)
        tensor_conversion = torchvision.transforms.ToTensor()
        image_tensor = tensor_conversion(padded_image)
        return image_tensor

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

        # set the padding to shift the start and end points coordinates
        self.pad_x1 = pad_x_1
        self.pad_y1 = pad_y_1

        # padding
        transform = torchvision.transforms.Pad((pad_x_1, pad_y_1, pad_x_2, pad_y_2))
        image_t = transform(image)

        return image_t

    @property
    def pad_x1(self):
        return self._pad_x1

    @pad_x1.setter
    def pad_x1(self, val):
        self._pad_x1 = val

    @property
    def pad_y1(self):
        return self._pad_y1

    @pad_y1.setter
    def pad_y1(self, val):
        self._pad_y1 = val

    def _get_points_tensor(self, data):
        pass


class WristPointsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, transform)
        self.heatmap = Heatmap()
        self.sigma = 5

    def __getitem__(self, idx):
        """
        :param idx: index of the item which we try to get the sample
        :return: sample: a dictionary consisting of image tensor, points tensor, patient_id and index
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inv_data_path = self.data[idx]

        #  Data path is different in remote server and local machine. line 46 for remote machine, 47 for local machine
        #  use the folder name as patient id

        patient_id = re.findall(r'\d+', inv_data_path.split("/")[1])[0]
        # patient_id = re.findall(r'\d+', inv_data_path.split("\\")[1])[0]
        with open(inv_data_path) as f:
            inv_data = json.load(f)
        self.image_tensor = self._get_image_tensor(item_path=inv_data_path)
        points_tensor = self._get_points_tensor(data=inv_data)
        points_heatmap_tensor = self._create_points_heatmap(point_tensor=points_tensor, sigma=self.sigma,
                                                            image_shape=self.image_shape)
        line_heatmap_tensor = self._create_line_heatmap(point_tensor=points_tensor, sigma=self.sigma,
                                                        image_shape=self.image_shape)

        sample = self.create_sample(self.image_tensor, points_tensor,
                                    points_heatmap_tensor, line_heatmap_tensor,
                                    patient_id, idx)

        # Data Augmentation
        if self.transform:
            sample = self.transform(sample)

        return sample

    def _get_points_tensor(self, data):
        """
                :param inv_data: json file of a specific data
                :return: points torch tensor -> shape: [1, 1, 4]
                """

        points = [d['points'] for d in data["shapes"] if d['label'] == 'Wrist Joint Line'][0]
        points = np.array([item for sublist in points for item in sublist])
        points = np.float32(points)

        # shift the coordinates regarding the padding size
        # start point
        points[0] += self.pad_x1
        points[1] += self.pad_y1

        # end point+
        points[2] += self.pad_x1
        points[3] += self.pad_y1

        points = np.expand_dims(np.asarray(points), axis=0)
        points = points[np.newaxis, :]
        points_tensor = torch.Tensor(points)

        return points_tensor

    def create_sample(self, image_tensor, points_tensor, heatmap_tensor,
                      heatmap_line, patient_id, index,  *args, **kwargs):
        sample = {'image': image_tensor, 'points': points_tensor,
                  'heatmap_points': heatmap_tensor, 'heatmap_line': heatmap_line,
                  'patient_id': patient_id, 'index': index}
        return sample

    def _create_points_heatmap(self, point_tensor, sigma, image_shape):
        points_heatmap_tensor = torch.zeros([2, self.image_shape, self.image_shape], dtype=torch.float32)
        x1, y1, x2, y2 = self.get_start_end_point(point_tensor)

        start_gauss = self.heatmap.apply_gauss(x1, y1, sigma, image_shape)
        start_gauss_tensor = torch.Tensor(start_gauss)
        points_heatmap_tensor[0] = start_gauss_tensor

        end_gauss = self.heatmap.apply_gauss(x2, y2, sigma, image_shape)
        end_gauss_tensor = torch.Tensor(end_gauss)
        points_heatmap_tensor[1] = end_gauss_tensor

        # display heatmap points
        # self.heatmap.display_heatmap(self.image_tensor, start_gauss, end_gauss)
        return points_heatmap_tensor

    def _create_line_heatmap(self, point_tensor, sigma, image_shape):
        line_heatmap_tensor = torch.zeros([1, self.image_shape, self.image_shape], dtype=torch.float32)
        x1, y1, x2, y2 = self.get_start_end_point(point_tensor)
        line_gauss = self.heatmap.line_heatmap(line_x=(y1, x1), line_y=(y2, x2), trunc=15,
                                               sigma=sigma, img_shape=image_shape)
        line_heatmap_tensor = line_gauss

        # display heatmap line
        # self.heatmap.display_heatmap_line(self.image_tensor, line_gauss)
        return line_heatmap_tensor

    def get_start_end_point(self, point_tensor):
        x1 = point_tensor.data[:, :, 0].item()
        y1 = point_tensor.data[:, :, 1].item()
        x2 = point_tensor.data[:, :, 2].item()
        y2 = point_tensor.data[:, :, 3].item()

        return x1, y1, x2, y2


class RadiusPointsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, transform)
        self._mask = None

    def _get_points_tensor(self, data):
        """
        :param inv_data: json file of a specific data
        :return: points numpy array -> shape: [47, 2]
        """
        points = [d['points'] for d in data["shapes"] if d['label'] == 'Radius'][0]
        points_arr = np.array(points, dtype='float32')
        points_arr[:, 0] += self.pad_x1
        points_arr[:, 1] += self.pad_y1
        return points_arr

    def create_sample(self, image_tensor, points_tensor, mask, patient_id, index):
        """

        :param image_tensor: size (512, 512)
        :param points_tensor: radius points: array (47, 2)
        :param mask: radius bone mask - tensor size (512, 512)
        :param patient_id: string
        :param index: int from 0- length of dataset
        :return: dictionary consisting of the above variables
        """
        sample = {'image': image_tensor, 'radius_points': points_tensor,
                  'radius': None, 'patient_id': patient_id, 'index': index}
        return sample

    def create_mask(self, points, img_shape):
        mask = np.zeros((img_shape, img_shape))
        rr, cc = polygon(points[:, 1], points[:, 0])
        rr = np.where(rr >= img_shape, img_shape-1, rr)
        cc = np.where(cc >= img_shape, img_shape-1, cc)
        mask[rr, cc] = 1
        mask = torch.from_numpy(mask)
        return mask

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, val):
        self._mask = val
