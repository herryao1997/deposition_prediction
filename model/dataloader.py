# -*- coding: utf-8 -*-
"""
dataloader.py

Load the dataset from the normalized JSON file created by preprocess.py.
This file defines the FluxDataset class, which is a PyTorch Dataset that
handles the loading of condition and output data for the training and
evaluation of machine learning models.

Author: Chengxi Yao
Email: stevenyao@g.skku.edu
"""

import json
import torch
from torch.utils.data import Dataset


class DepositDataset(Dataset):
    """
    A custom PyTorch Dataset class to load the normalized dataset from a JSON file.

    The dataset is expected to contain condition and output data, both of which are
    stored as normalized floating-point arrays. This class makes it easy to iterate
    through the dataset in batches during model training and evaluation.

    @param json_file Path to the normalized JSON file created by preprocess.py.
    """
    def __init__(self, json_file: str) -> None:
        """
        Initialize the dataset by loading data from the provided JSON file.

        @param json_file Path to the normalized JSON file.
        """
        super().__init__()

        # Load data from the normalized JSON file
        with open(json_file, 'r') as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        """
        Return the total number of data points in the dataset.

        This method is required by the PyTorch Dataset class.

        @return int The size of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Retrieve a single data point (condition and output) from the dataset by index.

        This method is required by the PyTorch Dataset class and is used to
        index into the dataset and return a specific sample. Both the condition
        and the output are returned as tensors.

        @param idx The index of the data point to retrieve.

        @return tuple A tuple containing:
            - condition_tensor (torch.Tensor): The condition vector.
            - output_tensor (torch.Tensor): The output vector (target).
        """
        # Get the data point at the given index
        item = self.data[idx]

        # Convert condition and output to PyTorch tensors
        cond = torch.tensor(item['condition'], dtype=torch.float32)
        out = torch.tensor(item['output'], dtype=torch.float32)

        # Uncomment these lines for debugging:
        # print(f"cond shape: {cond.shape}")
        # print(f"out shape: {out.shape}")

        return cond, out
