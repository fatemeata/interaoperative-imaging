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

    # ################################ image translation ###############################
    def get_translation(self, translation_vec):
        """
        :param: translation_vec: numpy vector
        :return: transformation matrix for translation
        """
        # tform = skimage.transform.EuclideanTransform(
        #     translation=translation_vec
        # )
        u = translation_vec[0]
        v = translation_vec[1]
        matrix = np.array([[1, 0, u],
                           [0, 1, v],
                           [0, 0, 1]])
        tform = skimage.transform.EuclideanTransform(matrix)
        # print(tform.params)
        return tform.params

    def transform(self, item, matrix):
        """
        :param item: dataloader item
        :param matrix: translation matrix
        :return: transformed item
        """
        # Apply translation matrix to dataloader item
        tform = skimage.transform.EuclideanTransform(
            matrix=matrix
        )
        # print(tform.params)
        tf_img = skimage.transform.warp(item, tform.inverse)
        return tf_img

    # ############################### image rotation #####################################
    def get_rotation(self, angle, img_shape_vec):
        """
        :param: theta: the given angel
        :return: transformation matrix around the center of image with given angel theta
        """
        # tform = skimage.transform.EuclideanTransform(
        #     rotation=math.radians(theta),
        # )
        # rotation_matrix = np.array([[np.cos(angle*(np.pi/180)), -np.sin(angle*(np.pi/180)), 0],
        #                    [np.sin(angle*(np.pi/180)), np.cos(angle*(np.pi/180)), 0],
        #                    [0, 0, 1]])
        rotation_matrix = transform.EuclideanTransform(rotation=math.radians(angle))
        shift_matrix = transform.EuclideanTransform(translation=-np.array(img_shape_vec) / 2)
        # Compose transforms by multiplying their matrices
        matrix = np.linalg.inv(shift_matrix.params) @ rotation_matrix.params @ shift_matrix.params
        tform = transform.EuclideanTransform(matrix)
        # print(tform.params)
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
        # print(tform.params)
        return tform.params

    # ############################## combined transformation #############################
    def get_transformation(self, s_val, r_val, t_val, img_shape):
        """
        :return: a combined transformation matrix of rotation, translation, and scaling
        """
        t_matrix = self.get_translation(t_val)
        r_matrix = self.get_rotation(r_val, img_shape)
        s_matrix = self.get_scaling(s_val)
        combined_matrix = np.matmul(t_matrix, r_matrix, s_matrix)
        return combined_matrix

    def transform_points(self, points, matrix):
        transformed_vec = np.matmul(points, matrix.T)
        return transformed_vec

    def perform_augmentation(self, img_shape):
        # randomly chooses a translation vector in range of (min, max) translation
        # rotation vector in range of (min, max) rotation
        # scaling value in range of (min, max) scaling
        rand_scale_val = np.random.uniform(self.min_scaling, self.max_scaling)
        rand_angle_val = np.random.uniform(self.min_rotation, self.max_rotation)
        rand_x_val = np.random.uniform(self.min_translation, self.max_translation)
        rand_y_val = np.random.uniform(self.min_translation, self.max_translation)
        rand_translation = (rand_x_val, rand_y_val)

        print("random angle, scale, translation: ", rand_angle_val, rand_scale_val, rand_translation)
        matrix = self.get_transformation(rand_scale_val, rand_angle_val, rand_translation, img_shape)

        return matrix
