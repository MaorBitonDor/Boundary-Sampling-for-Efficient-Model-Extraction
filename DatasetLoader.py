import os

import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetLoader(Dataset):
    """
    Initialize the DatasetLoader. This custom dataloader is used to load the data that was saved to the disk during the
    evolutionary algorithm process. This dataloader doesn't load all the dataset to the memory to able the code to run on
    machines with less memory.

    :param data_dir: The directory containing the data files.
    :param file_size: The number of samples per file.
    :param transform: A function/transform to apply to the data.
    :param subset_generations: The number of generations to include from each subset.
        If None, all generations are included.
    :return: None
    """
    def __init__(self, data_dir, file_size=20000, transform=None, subset_generations=None):
        self.data_dir = data_dir
        self.transform = transform
        file_list = os.listdir(data_dir)
        self.data_list = sorted([x for x in file_list if "_input" in x])
        self.labels_list = sorted([x for x in file_list if "_labels" in x])
        if subset_generations is not None:
            away_data = sorted([x for x in self.data_list if "Away_" in x], key=lambda y: int(y.split("_")[-2]))
            away_labels = sorted([x for x in self.labels_list if "Away_" in x], key=lambda y: int(y.split("_")[-2]))
            toward_data = sorted([x for x in self.data_list if "Toward_" in x], key=lambda y: int(y.split("_")[-2]))
            toward_labels = sorted([x for x in self.labels_list if "Toward_" in x], key=lambda y: int(y.split("_")[-2]))
            self.data_list = away_data[:subset_generations] + toward_data[:subset_generations]
            self.labels_list = away_labels[:subset_generations] + toward_labels[:subset_generations]
        self.file_size = file_size

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        :return: The total number of samples.
        """
        return len(self.data_list) * self.file_size

    def __getitem__(self, idx: int) -> tuple[any, any]:
        """
        Get a sample from the dataset at the specified index.

        :param idx: The index of the sample to retrieve.
        :return: A tuple containing the data sample and its corresponding label.
        """
        file_idx = idx // self.file_size  # Calculate which file to load
        sample_idx = idx % self.file_size  # Calculate the index within the file

        data_file_name = self.data_list[file_idx]
        labels_file_name = self.labels_list[file_idx]
        data_file_path = os.path.join(self.data_dir, data_file_name)
        labels_file_path = os.path.join(self.data_dir, labels_file_name)

        # Load the data tensor from the file
        data = np.load(data_file_path, mmap_mode='r')  # Use mmap_mode to avoid loading the entire file into memory
        labels = np.load(labels_file_path, mmap_mode='r')  # Use mmap_mode to avoid loading the entire file into memory

        # Extract the individual tensor at the specified index
        data = data[sample_idx]
        labels = labels[sample_idx]

        # You can apply transformations here if needed
        if self.transform:
            # Convert to PIL Image
            data_size = data.shape
            pil_image = torch.tensor(data.squeeze(0)).permute(1, 2, 0).numpy()
            pil_image = self.transform(pil_image)
            pil_image.view(data_size)
            # Convert back to tensor
            data = pil_image

        return data, labels

    def set_transform(self, transform: callable) -> None:
        """
        Set the transformation to be applied to the data.

        :param transform: The transformation function to apply.
        :return: None
        """
        self.transform = transform
