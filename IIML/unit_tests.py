import os
import skimage
import dataset
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw


class UnitTests:
    def __init__(self):
        self.root_dir = "data"
        self.dataset = dataset.WristDataset(root_dir=self.root_dir)
        self.data = []

    def show_image(self, idx):
        item = self.dataset[idx]
        image_tensor = item['image']
        image_tensor = torch.squeeze(image_tensor, 0)

        point_tensor = item['points']
        self.display_points_on_image(image_tensor, point_tensor)

    def display_points_on_image(self, image_tensor, point_tensor):
        """
        :param image_tensor: image torch tensor
        :param point_tensor: points torch tensor
        :return: none -> display the start point and end point on the image
        """

        x1 = point_tensor.data[:, :, 0].item()
        y1 = point_tensor.data[:, :, 1].item()
        shape_1 = [(x1-2, y1-2), (x1+2, y1+2)]

        x2 = point_tensor.data[:, :, 2].item()
        y2 = point_tensor.data[:, :, 3].item()
        shape_2 = [(x2 - 2, y2 - 2), (x2 + 2, y2 + 2)]

        print("start point: ", x1, y1)
        print("end point: ", x2, y2)
        img = T.ToPILImage()(image_tensor)

        draw = ImageDraw.Draw(img)
        draw.ellipse(shape_1, fill="#fff")
        draw.ellipse(shape_2, fill="#fff")

        img.show()


if __name__ == "__main__":
    unit_test = UnitTests()
    print("NUMBER OF IMAGES: ", len(unit_test.dataset))
    unit_test.show_image(idx=165)
