# -*- coding: utf-8 -*-
"""
config_csv_json.py

This script constructs the folder structure for train and validation datasets,
and creates JSON files with metadata regarding the prepared CSV files. It reads
CSV files, splits them into training and validation sets, copies them into
corresponding directories, and stores metadata in a JSON file.

Author: Chengxi Yao
Email: stevenyao@g.skku.edu
"""

__author__ = "Chengxi Yao"
__email__ = "stevenyao@g.skku.edu"

import os
import json
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split


def create_json_and_split(csv_files: list, train_ratio: float, output_dir: str) -> None:
    """
    Create JSON file and split CSV files into training and validation sets.

    This function takes a list of CSV files, splits them into training and validation
    sets based on the provided train/validation ratio, and saves them into separate
    directories. It also generates a JSON file that contains metadata about each file,
    including the filename, its split (train/valid), and a parsed condition extracted
    from the 10th column of the CSV.

    Args:
        csv_files (list): List of paths to CSV files to be split.
        train_ratio (float): The ratio of the dataset to be used for training.
        output_dir (str): The output directory where the train/valid datasets and JSON file will be saved.
    """
    train_dir = os.path.join(output_dir, 'train')
    valid_dir = os.path.join(output_dir, 'valid')

    # Create directories for train and valid data if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    # Split the provided CSV files into train and valid sets
    train_files, valid_files = train_test_split(csv_files, train_size=train_ratio, random_state=42)

    # Initialize an empty list to store metadata about each file
    data = []

    def process_files(files, split, target_dir):
        """
        Helper function to process and copy files to the target directory.

        This function reads each file, extracts its metadata (such as filename and
        condition), and copies the file into the corresponding train/valid folder.

        Args:
            files (list): List of file paths to process.
            split (str): The split type ('train' or 'valid').
            target_dir (str): The directory to copy the processed files into.
        """
        for file_path in files:
            file_info = {}
            file_name = os.path.basename(file_path)  # Get the base name of the file
            file_info['file_name'] = file_name  # Store just the file name
            file_info['split'] = split  # Indicate if it's train or valid

            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path, encoding='ISO-8859-1')  # Handling special character encoding
            df.columns = df.columns.str.strip()  # Clean the column names by removing any spaces

            # Extract the condition from the 10th column (index 9) if it exists
            if len(df.columns) > 9:
                condition_str = df.columns[9]  # 10th column name
            else:
                print(f"Error: Not enough columns in file {file_name}")
                condition_str = None  # Handle missing columns

            # Parse the condition string into a tuple of integers if possible
            if condition_str is not None:
                condition_str = str(condition_str)  # Ensure it's a string
                try:
                    condition_tuple = tuple(map(int, condition_str.split('-')))
                except ValueError:
                    print(f"Error parsing condition in file {file_name}: {condition_str}")
                    condition_tuple = None  # Handle parsing errors
            else:
                condition_tuple = None

            # Store the parsed condition in the metadata
            file_info['condition'] = condition_tuple

            # Copy the CSV file into the respective train/valid folder
            shutil.copy(file_path, os.path.join(target_dir, file_name))

            # Append the metadata to the dataset
            data.append(file_info)

    # Process and copy the files to the train and valid directories
    process_files(train_files, 'train', train_dir)
    process_files(valid_files, 'valid', valid_dir)

    # Save the metadata to a JSON file
    with open(os.path.join(output_dir, 'file_metadata.json'), 'w') as f:
        json.dump(data, f, indent=4)

"""
# Example usage:
# If you have a list of CSV files in a folder 'data_base', you can use the following:
csv_files = [f"data_base/file_{i}.CSV" for i in range(48)]  # Adjust the file paths as needed
create_json_and_split(csv_files, train_ratio=0.8, output_dir='data/split_data')
"""
csv_files = [f"data_base/file_{i}.CSV" for i in range(1, 48)]  # Adjust paths as needed
create_json_and_split(csv_files, train_ratio=0.8, output_dir='data/split_data')
