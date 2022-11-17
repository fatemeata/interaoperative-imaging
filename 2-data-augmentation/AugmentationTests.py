import unittest

import torch

import intensity_augmentation
import geometry_augmentation
import math
from dataset import WristDataset
import torchvision.transforms as torch_transform
import matplotlib.pyplot as plt
import numpy as np
from unit_tests import UnitTests
from PIL import Image
import skimage


class AugmentationTest(unittest.TestCase):

    def setUp(self) -> None:
        root_dir = "data"
        self.dataset = WristDataset(root_dir)
        super().__init__()
        # geometry and intensity instances
        self.geometry = geometry_augmentation.GeometryAugmentation()
        self.intensity = intensity_augmentation.IntensityAugmentation()

        # get the image and point tensor of given index
        self.index = 0
        self.original_image_tensor = self.dataset[self.index]["image"]
        self.original_point_tensor = self.dataset[self.index]["points"]

        x_s, y_s = np.asarray(self.original_point_tensor[0])
        self.original_start_point = [x_s, y_s]
        x_e, y_e = np.asarray(self.original_point_tensor[1])
        self.original_end_point = [x_e, y_e]

        # convert image to numpy array
        self.img = np.array(torch_transform.ToPILImage()(self.original_image_tensor))
        self.points = self.raw_points_process(self.original_point_tensor)

        # image shape required for rotation around the center
        self.img_shape = self.img.shape[:2]  # (512, 512)

    def test_translation(self):
        translation = (50, 100)
        translation_matrix = self.geometry.get_translation(translation)
        translated_img = np.float32(self.geometry.transform(item=self.img, matrix=translation_matrix))
        intensity_scaled_image = skimage.exposure.rescale_intensity(translated_img)
        translated_points = self.point_translation_process(point_tensor=self.points, matrix=translation_matrix)

        print("translated points: ", translated_points)
        self.visualize_data(t_image=intensity_scaled_image, t_points=translated_points)

    def test_rotation(self):
        theta = 45

        rotation_matrix = self.geometry.get_rotation(angle=theta, img_shape_vec=self.img_shape)
        rotated_img = self.geometry.transform(item=self.img, matrix=rotation_matrix)
        rotated_points = self.point_translation_process(point_tensor=self.points, matrix=rotation_matrix)
        # rotated_points = self.point_rotation_process(point_tensor=self.points, theta=theta)
        self.visualize_data(t_image=rotated_img, t_points=rotated_points)

    def test_scaling(self):
        scale_value = 0.5
        # scale_value = 1.2
        scale_matrix = self.geometry.get_scaling(scale_value)
        scaled_image = self.geometry.transform(item=self.img, matrix=scale_matrix)
        scaled_points = self.point_translation_process(point_tensor=self.points, matrix=scale_matrix)
        self.visualize_data(t_image=scaled_image, t_points=scaled_points)

    def test_combined_transformation(self):
        scale_val = 0.8
        rotate_angle = 45
        translation_mat = (50, 100)
        comb_transform_matrix = self.geometry.get_transformation(s_val=scale_val, r_val=rotate_angle,
                                                                 t_val=translation_mat, img_shape=self.img_shape)
        transformed_image = self.geometry.transform(item=self.img, matrix=comb_transform_matrix)
        transformed_points = self.point_translation_process(point_tensor=self.points, matrix=comb_transform_matrix)
        self.visualize_data(t_image=transformed_image, t_points=transformed_points)

    def test_random_transformation(self):
        rand_transform_matrix = self.geometry.perform_augmentation(self.img_shape)
        rand_transformed_image = self.geometry.transform(item=self.img, matrix=rand_transform_matrix)
        rand_transformed_points = self.point_translation_process(point_tensor=self.points, matrix=rand_transform_matrix)
        self.visualize_data(t_image=rand_transformed_image, t_points=rand_transformed_points)

    def raw_points_process(self, raw_points):
        points = np.array(raw_points)
        points = np.expand_dims(np.asarray(points), axis=2)
        expanded_points = np.insert(points, points.shape[1], 1, axis=1)
        return expanded_points

    def point_translation_process(self, point_tensor, matrix):
        translated_points_list = []
        for i in range(point_tensor.shape[0]):
            transformed_vec = self.geometry.transform_points(
                points=point_tensor[i].T, matrix=matrix)
            translated_points_list.append(transformed_vec[:, 0:2])
        translated_points = np.reshape(np.array(translated_points_list), (2, 2))
        return translated_points

    def test_contrast_change(self):
        contrast_img = self.intensity.contrast_change(s_val=1.4, item=self.img)
        self.visualize_data(t_image=contrast_img, t_points=None)

    def test_inversion(self):
        inverted_img = self.intensity.invert(item=self.img)
        self.visualize_data(t_image=inverted_img, t_points=None)

    def test_normalization(self):
        normalized_img = self.intensity.normalize(item=self.img)
        self.visualize_data(t_image=normalized_img, t_points=None)

    def test_perform_augmentation(self):
        augmented_img = self.intensity.perform_augmentation(img=self.img)
        self.visualize_data(t_image=augmented_img, t_points=None)

    def visualize_data(self, t_image, t_points):

        if t_points is not None:
            x1, y1 = t_points[0]
            x2, y2 = t_points[1]
        else:
            # for intensity augmentation
            x1, y1 = self.original_start_point[0], self.original_start_point[1]
            x2, y2 = self.original_end_point[0], self.original_end_point[1]

        f, ax_arr = plt.subplots(1, 2)

        ax_arr[0].imshow(self.img, interpolation='nearest', cmap='gray')
        # ax_arr[0].plot(self.original_start_point, self.original_end_point, color="white", linewidth=2)
        ax_arr[0].plot(self.original_start_point[0], self.original_start_point[1], marker='v')
        ax_arr[0].plot(self.original_end_point[0], self.original_end_point[1], marker='v')

        ax_arr[1].imshow(t_image, interpolation='nearest', cmap='gray')
        ax_arr[1].plot(x1, y1, marker='v')
        ax_arr[1].plot(x2, y2, marker='v')

        # plt.gray()
        plt.show()


