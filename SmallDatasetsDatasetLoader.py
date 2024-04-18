import os

import numpy as np
import torch
from torch.utils.data import Dataset


class SmallDatasetDatasetLoader(Dataset):
    def __init__(self, data_dir: str) -> None:
        """
        Initialize the SmallDatasetDatasetLoader. This custom dataloader is used to load the data that was saved to the
        disk during the evolutionary algorithm process and loading it into the memory to a list for fast access.

        :param data_dir: The directory containing the data files.
        :return: None
        """
        # Convert the list of tuples to a list of tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Load the data tensor from the file
        self.data_dir = data_dir
        file_list = os.listdir(self.data_dir)
        data_list = sorted([x for x in file_list if "_input" in x])
        labels_list = sorted([x for x in file_list if "_labels" in x])
        self.tensor_list = []
        self.label_list = []
        for data_file_name, labels_file_name in zip(data_list, labels_list):
            data_file_path = os.path.join(self.data_dir, data_file_name)
            labels_file_path = os.path.join(self.data_dir, labels_file_name)
            data = np.load(data_file_path, mmap_mode='r')
            labels = np.load(labels_file_path, mmap_mode='r')
            data = torch.split(torch.tensor(data), split_size_or_sections=1, dim=0)
            labels = list(torch.split(torch.tensor(labels), split_size_or_sections=1, dim=0))
            shape = data[0].shape
            shape_labels = labels[0].shape
            data = [x.view(-1, shape[2], shape[3], shape[4]) for x in data]
            labels = [x.view(-1, shape_labels[2]) for x in labels]
            self.tensor_list.extend(data)
            self.label_list.extend(labels)

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        :return: The total number of samples.
        """
        return len(self.tensor_list)

    def add_data(self, cur_tensor_list: list[torch.Tensor], cur_label_list: list[torch.Tensor]) -> None:
        """
        Add additional data to the dataset.

        :param cur_tensor_list: The list of tensor data to add.
        :param cur_label_list: The list of corresponding labels to add.
        :return: None
        """
        self.tensor_list.extend(cur_tensor_list)
        self.label_list.extend(cur_label_list)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset at the specified index.

        :param idx: The index of the sample to retrieve.
        :return: A tuple containing the data sample and its corresponding label.
        """
        data, label = self.tensor_list[idx], self.label_list[idx]

        return data, label
