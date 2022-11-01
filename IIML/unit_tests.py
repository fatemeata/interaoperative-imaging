import os
import skimage
import dataset
import matplotlib as plt
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image, ImageDraw

class UnitTests:
    def __init__(self):
        self.root_dir = "data"
        self.dataset = dataset.WristDataset(root_dir=self.root_dir)
        self.data = []

    def dataset_iteration(self):
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            self.data.append(item)
            #print(f"{i}- ", item['image'].shape, item['points'].shape)

    def dataset_len(self):
        print("dataset length: ", len(self.dataset))

    def show_image(self, idx):
        item = self.dataset[idx]
        image_tensor = item['image']
        image_tensor = torch.squeeze(image_tensor, 0)

        point_tensor = item['points']
        print("point tensor: ", point_tensor)
        x_1 = point_tensor.data[:, :, 0].item()
        y_1 = point_tensor.data[:, :, 1].item()
        shape_1 = [(x_1-2, y_1-2), (x_1+2, y_1+2)]

        x_2 = point_tensor.data[:, :, 2].item()
        y_2 = point_tensor.data[:, :, 3].item()
        shape_2 = [(x_2-2, y_2-2), (x_2+2, y_2+2)]

        transform = T.ToPILImage()
        img = transform(image_tensor)
        draw = ImageDraw.Draw(img)
        draw.ellipse(shape_1, fill="#fff")
        draw.ellipse(shape_2, fill="#fff")
        img.show()
        img = img.save("joint.png")


if __name__ == "__main__":
    unit_test = UnitTests()
    # unit_test.dataset_iteration()
    unit_test.show_image(idx=100)
