import os
import numpy as np
import skimage
import torchvision
from PIL import Image


class IntensityAugmentation:

    def __init__(self):
        self.min_contrast_change = 0.5
        self.max_contrast_change = 2
        self.prob_inversion = 0.5

    def contrast_change(self, s_val, item):
        """
        :param s_val: scalar
        :param item: item from dataloader: image
        :return: modified item
        """
        m = np.mean(item)
        v_prime = (s_val*(item-m) + m).astype('uint8')
        v_prime = np.float32(v_prime)
        return v_prime

    def invert(self, item):
        """
        :param item: item from dataloader
        :return: item which values are being inverted
        """
        # img = Image.fromarray(item)
        inverted_img = 255 - item
        return inverted_img

    def normalize(self, item):
        """
        :param item: item from dataloader
        :return: normalized item
        """
        normalized_item = None
        item_mean = np.mean(item)
        item_dev = np.std(item)
        normalized_item = (item - item_mean) / item_dev
        # # check the item deviation is not zero
        # if item_dev != 0:
        # else:
        #     print("item deviation is zero. check the image value! ")

        return normalized_item

    def perform_augmentation(self, img):
        """
        :param img: original image (array -> float32)
        :return: modified image (apply contrast, inversion, and normalization)
        """
        # chooses a random contrast value in range of min and max
        # chooses a random inversion value in range of min and max

        rand_contrast_val = np.random.uniform(self.min_contrast_change, self.max_contrast_change)
        rand_inversion_val = np.random.uniform(0, 1)

        changed_img = self.contrast_change(s_val=rand_contrast_val, item=img)
        if rand_inversion_val < self.prob_inversion:
            changed_img = self.invert(item=changed_img)
        # normalized_img = self.normalize(item=changed_img)
        return changed_img

