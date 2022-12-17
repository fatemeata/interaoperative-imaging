import unittest
import dataset
from utils.data_augmentation import RadiusDataAugmentation, WristLineDataAugmentation


class SegmentTest(unittest.TestCase):
    def setUp(self) -> None:
        root_dir = "data"
        self.wrist_data = dataset.WristPointsDataset(root_dir=root_dir, transform=WristLineDataAugmentation())
        self.radius_data = dataset.RadiusPointsDataset(root_dir=root_dir, transform=RadiusDataAugmentation())

    def test_radius_dataset(self):
        index = 0
        # first_sample_wrist = self.wrist_data[index]
        first_sample_radius = self.radius_data[index]

