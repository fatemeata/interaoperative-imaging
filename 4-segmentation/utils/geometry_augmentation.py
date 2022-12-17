import skimage
import numpy as np
import math
from skimage import transform
import random

class GeometryAugmentation:

    def __init__(self):
        self.min_translation = -100
        self.max_translation = 100
        self.min_rotation = -45
        self.max_rotation = 45
        self.min_scaling = 0.8
        self.max_scaling = 1.5

    def transform(self, item, matrix):
        """
        takes an image and transform matrix and returns the transformed image
        :param item: dataloader item
        :param matrix: translation matrix
        :return: transformed item
        """
        # Apply translation matrix to dataloader item
        tform = skimage.transform.EuclideanTransform(
            matrix=matrix
        )
        tf_img = skimage.transform.warp(item, tform.inverse)
        return tf_img

    # ################################ image translation ###############################
    def get_translation(self, translation_vec):
        """
        provides translation matrix
        :param: translation_vec: numpy vector
        :return: transformation matrix for translation
        """
        u = translation_vec[0]
        v = translation_vec[1]
        matrix = np.array([[1, 0, u],
                           [0, 1, v],
                           [0, 0, 1]])
        tform = skimage.transform.EuclideanTransform(matrix)
        return tform.params

    # ############################### image rotation #####################################
    def get_rotation(self, angle, img_shape_vec):
        """
        :param: angle: the given angel
        :param: img_shape_vec: image size to provide the center of image
        :return: transformation matrix around the center of image with given angel theta
        """
        rotation_matrix = transform.EuclideanTransform(rotation=math.radians(angle))
        shift_matrix = transform.EuclideanTransform(translation=-np.array(img_shape_vec) / 2)
        # Compose transforms by multiplying their matrices
        matrix = np.linalg.inv(shift_matrix.params) @ rotation_matrix.params @ shift_matrix.params
        tform = transform.EuclideanTransform(matrix)
        return tform.params

    # ############################### image scaling ######################################
    def get_scaling(self, scale_val):
        """
        :param: scale_val: scale value scalar
        :return: transformation matrix for scaling image
        """
        matrix = np.array([[scale_val, 0, 0],
                           [0, scale_val, 0],
                           [0, 0, 1]])
        tform = skimage.transform.EuclideanTransform(matrix)
        return tform.params

    # ############################## combined transformation #############################
    def get_transformation(self, s_val, r_val, t_val, img_shape):
        """

        :param s_val: scaling value
        :param r_val: rotation angle value
        :param t_val: translation vector
        :param img_shape: image size to provide the center of the image for rotation matrix
        :return: a combined transformation matrix of rotation, translation, and scaling
        """
        t_matrix = self.get_translation(t_val)
        r_matrix = self.get_rotation(r_val, img_shape)
        s_matrix = self.get_scaling(s_val)
        combined_matrix = np.matmul(t_matrix, r_matrix, s_matrix)
        return combined_matrix

    def transform_points(self, points, matrix):
        """
        transform the points given a transformation (rotation, scaling or translation) matrix
        :param points: point tensor
        :param matrix: transformation matrix
        :return: transformed point tensor vector
        """
        transformed_vec = np.matmul(points, matrix.T)
        return transformed_vec

    def point_translation_process(self, point_tensor, matrix):
        point_tensor = self._reshape_points(point_tensor)
        translated_points_list = []
        for i in range(point_tensor.shape[0]):
            transformed_vec = self.transform_points(
                points=point_tensor[i].T, matrix=matrix)
            translated_points_list.append(transformed_vec[:, 0:2])
        translated_points = np.reshape(np.array(translated_points_list), (2, 2))
        return translated_points

    def perform_augmentation(self, img_shape):
        """
        randomly chooses a translation vector in range of (min, max) translation
        - rotation vector in range of (min, max) rotation
        - scaling value in range of (min, max) scaling
        :param img_shape: image size to provide the center of image for rotation matrix
        :return: the transformation matrix
        """

        rand_scale_val = np.random.uniform(self.min_scaling, self.max_scaling)
        rand_angle_val = np.random.uniform(self.min_rotation, self.max_rotation)
        rand_x_val = np.random.uniform(self.min_translation, self.max_translation)
        rand_y_val = np.random.uniform(self.min_translation, self.max_translation)
        rand_translation = (rand_x_val, rand_y_val)
        matrix = self.get_transformation(rand_scale_val, rand_angle_val, rand_translation, img_shape)

        return matrix

    def _reshape_points(self, points):
        points = points.numpy()
        points = np.reshape(points, (2, 2))  # shape -> (2, 2)
        points = self.raw_points_process(points)  # shape -> (2, 2, 1) it would be like that: (x1, y1, 1) - (x2, y2, 1)
        points = np.float32(points)
        return points

    def raw_points_process(self, raw_points):
        points = np.array(raw_points)
        points = np.expand_dims(np.asarray(points), axis=2)
        expanded_points = np.insert(points, points.shape[1], 1, axis=1)
        return expanded_points

    def radius_points_translation_process(self, point_tensor, matrix):
        point_tensor = self.raw_points_process(point_tensor)
        translated_points = []
        # the error may happen here!
        for i in range(point_tensor.shape[0]):
            transformed_vec = self.transform_points(
                points=point_tensor[i].T, matrix=matrix)
            translated_points.append(transformed_vec[:, 0:2])
        translated_points = np.reshape(np.array(translated_points), (point_tensor.shape[0], 2))
        return translated_points
