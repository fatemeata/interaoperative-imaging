import os

import numpy as np
import torch
import torchvision
from dataset import RadiusPointsDataset, WristPointsDataset
import itertools


class DataSplitting:

    def __init__(self):
        self.data = []  # the whole dataset

    def random_splitting(self, dataset):
        """
        :param dataset: dataset
        :return: train, validation set and test set
        """
        # train, validation, test: 0.6 / 0.2 / 0.2
        split_dataset = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2])
        train = split_dataset[0]
        validation = split_dataset[1]
        test = split_dataset[2]
        return train, validation, test

    def k_fold_data(self, data):
        """
        :param data: could be wrist dataset
        :return:
        """
        patient_list = self.patient_id_list(data)
        unique_patient_list = list(set(patient_list))

        # split data to 5 folds
        folded_p_list = self.k_fold_splitting(k_fold=5, p_list=unique_patient_list)

        # creates a subset for training, validation and testing.
        train, validation, test = self.create_subset(dataset=self.data, folded_dataset=folded_p_list)
        return train, validation, test

    def patient_id_list(self, dataset):
        pid_list = []
        for idx in range(len(dataset)):
            self.data.append(dataset[idx])
            pid = dataset[idx]["patient_id"]
            pid_list.append(pid)
        return pid_list

    def k_fold_splitting(self, k_fold, p_list):
        """
        :param k_fold: number of folds
        :param p_list: list of all patient id
        :return:  (nested) list of length of folds, containing a list with the item ids
        """
        folded_list = list(np.array_split(np.array(p_list), k_fold))
        return folded_list

    def create_subset(self, dataset, folded_dataset):
        """
        :param dataset: the whole dataset
        :param folded_dataset: the folded_dataset
        :return: list [train_set, validation_set, test_set]
        """
        validation = folded_dataset[0]
        v_indices = list(itertools.chain(*self.get_set_indices(validation)))  # flatten the list

        test = folded_dataset[1]
        te_indices = list(itertools.chain(*self.get_set_indices(test)))

        train = list(np.concatenate(folded_dataset[2:]))
        tr_indices = list(itertools.chain(*self.get_set_indices(train)))

        validation_set = torch.utils.data.Subset(dataset, v_indices)
        test_set = torch.utils.data.Subset(dataset, te_indices)
        train_set = torch.utils.data.Subset(dataset, tr_indices)

        return train_set, validation_set, test_set

    def get_patient_indices(self, pid):
        """
        :param pid: patient id
        :return: list of all image indices of one patient with given pid
        """
        return [item["index"] for item in self.data if item["patient_id"] == pid]

    def get_set_indices(self, p_set):
        """
        :param p_set: list of patient in a set
        :return: list of all indices for all patient in a set
        """
        return [self.get_patient_indices(item) for item in p_set]
