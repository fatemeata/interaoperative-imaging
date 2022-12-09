import torch.nn as nn
import torchvision
from .geometry_augmentation import GeometryAugmentation
from .intensity_augmentation import IntensityAugmentation
import numpy as np
import torch
import torchvision.transforms as torch_transform
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt


class DataAugmentation(nn.Module):

    def __init__(self):
        super().__init__()
        self.geometry = GeometryAugmentation()
        self.intensity = IntensityAugmentation()
        self.img_shape = (512, 512)

    def forward(self, sample):
        img = sample['image']
        points = sample['points']
        # self.display_img_points(img, points)
        img = np.array(torch_transform.ToPILImage()(img)).astype(dtype="float64")
        t_img, t_points = self._geometry(img, points)
        points_tensor = self._reshape_points_tensor(t_points)
        tensor_conversion = torchvision.transforms.ToTensor()
        image_tensor = tensor_conversion(t_img)
        sample['image'] = image_tensor
        sample['points'] = points_tensor

        # self.display_img_points(image_tensor, points_tensor)
        return sample

    def _geometry(self, img, points):
        rand_transform_matrix = self.geometry.perform_augmentation(self.img_shape)
        rand_transformed_image = np.float32(self.geometry.transform(item=img, matrix=rand_transform_matrix))
        rand_transformed_points = self.geometry.point_translation_process(point_tensor=points,
                                                                          matrix=rand_transform_matrix)
        return rand_transformed_image, rand_transformed_points

    def _intensity(self, img):
        augmented_img = self.intensity.perform_augmentation(img)
        return augmented_img

    def _reshape_points_tensor(self, points):
        points = np.reshape(points, (1, 4))
        points = points[np.newaxis, :]
        points = np.float32(points)
        points_tensor = torch.Tensor(points)
        return points_tensor

    def display_img_points(self, image_tensor, point_tensor):
        """
        display points on the image, can be called before and after augmentation
        :param image_tensor: image tensor
        :param point_tensor:
        :return:
        """
        plt.imshow(image_tensor.permute(1, 2, 0), cmap="gray")
        x1 = point_tensor.data[:, :, 0].item()
        y1 = point_tensor.data[:, :, 1].item()
        x2 = point_tensor.data[:, :, 2].item()
        y2 = point_tensor.data[:, :, 3].item()
        plt.plot(x1, y1, marker='.')
        plt.plot(x2, y2, marker='.')
        plt.show()
