import unittest
import os
import torch
import numpy as np
from dataset import WristDataset
from data_splitting import DataSplitting


class DatasetTest(unittest.TestCase):

    def setUp(self) -> None:
        super().__init__()
        root_dir = "data"
        self.dataset = WristDataset(root_dir)
        self.data_split = DataSplitting()

    def test_random_splitting(self):
        train, validation, test = self.data_split.random_splitting(self.dataset)
        print(f"train set size: {len(train)},\nvalidation set size: {len(validation)},\ntest set size: {len(test)}")

        # **************************** Problem statement ********************************
        # Our dataset consists of different patients and every patient may have different images.
        # if we split the data randomly, it's a possibility to have one patient images in different train, validation,
        # or test set, which ignore the underlying concept of splitting data

    def test_k_fold_splitting(self):
        # init dataset
        patient_list = self.data_split.patient_id_list(self.dataset)
        unique_patient_list = list(set(patient_list))

        # split data to 5 folds
        folded_p_list = self.data_split.k_fold_splitting(k_fold=5, p_list=unique_patient_list)

        # creates a subset for training, validation and testing.
        train, validation, test = self.data_split.create_subset(folded_p_list)
        print(f"length of train set: {len(train)}, "
              f"validation set: {len(validation)},"
              f" test set: {len(test)}")
