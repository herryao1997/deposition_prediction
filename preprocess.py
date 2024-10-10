# -*- coding: utf-8 -*-
"""
preprocess.py

This script preprocesses CSV data by normalizing the condition and output
vectors for train and valid datasets, and saves all data into single JSON files.

Author: Chengxi Yao
Email: stevenyao@g.skku.edu
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils import Params
import joblib  # Required for saving and loading scalers if scikit-learn version >= 0.23


def preprocess_data(metadata_file: str, data_dir: str, params: Params):
    """
    Preprocess CSV data by normalizing the condition and output vectors, and save to JSON files.

    This function loads metadata, reads the corresponding CSV files, and extracts the condition
    and output vectors for both training and validation datasets. It then normalizes the data
    using StandardScaler and saves the preprocessed data and the scalers to disk.

    @param metadata_file Path to the metadata JSON file containing file paths and conditions.
    @param data_dir Directory containing the 'train' and 'valid' folders with CSV data.
    @param params Model parameters that include the output dimension and other settings.

    @return None
    """
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Separate metadata into train and valid datasets
    train_metadata = [item for item in metadata if item['split'] == 'train']
    valid_metadata = [item for item in metadata if item['split'] == 'valid']

    # Initialize lists to store conditions and outputs for training data
    train_conditions = []
    train_outputs = []
    train_data = []

    # Process each item in the training metadata
    for item in train_metadata:
        file_name = item['file_name']
        condition = item['condition']
        file_path = os.path.join(data_dir, 'train', file_name)

        # Read outputs from CSV file
        data = pd.read_csv(file_path, encoding='ISO-8859-1')
        data.columns = data.columns.str.strip()  # Clean up column names

        # Extract the 'outputs' column and ensure output dimension matches params
        if 'outputs' in data.columns:
            outputs = data['outputs'].values[:params.output_dim]
        else:
            print(f"Error: 'outputs' column not found in file {file_name}")
            continue

        # Append the condition and outputs to respective lists
        train_conditions.append(condition)
        train_outputs.append(outputs)
        train_data.append({'condition': condition, 'output': outputs})

    # Fit scalers on training data
    scaler_condition = StandardScaler()
    scaler_output = StandardScaler()

    # Fit the scaler on the condition data
    scaler_condition.fit(train_conditions)

    # Flatten all outputs to fit the scaler
    all_train_outputs = np.concatenate(train_outputs).reshape(-1, 1)
    scaler_output.fit(all_train_outputs)

    # Normalize the training data using the fitted scalers
    normalized_train_data = []
    for sample in train_data:
        condition_normalized = scaler_condition.transform([sample['condition']])[0].tolist()
        outputs_normalized = scaler_output.transform(sample['output'].reshape(-1, 1)).flatten().tolist()

        normalized_data = {
            'condition': condition_normalized,
            'output': outputs_normalized
        }
        normalized_train_data.append(normalized_data)

    # Process validation data following the same steps as training data
    valid_conditions = []
    valid_outputs = []
    valid_data = []

    for item in valid_metadata:
        file_name = item['file_name']
        condition = item['condition']
        file_path = os.path.join(data_dir, 'valid', file_name)

        # Read outputs from CSV file
        data = pd.read_csv(file_path, encoding='ISO-8859-1')
        data.columns = data.columns.str.strip()  # Clean up column names

        # Extract the 'outputs' column and ensure output dimension matches params
        if 'outputs' in data.columns:
            outputs = data['outputs'].values[:params.output_dim]
        else:
            print(f"Error: 'outputs' column not found in file {file_name}")
            continue

        # Append the condition and outputs to respective lists
        valid_conditions.append(condition)
        valid_outputs.append(outputs)
        valid_data.append({'condition': condition, 'output': outputs})

    # Normalize validation data using the training scalers
    normalized_valid_data = []
    for sample in valid_data:
        condition_normalized = scaler_condition.transform([sample['condition']])[0].tolist()
        outputs_normalized = scaler_output.transform(sample['output'].reshape(-1, 1)).flatten().tolist()

        normalized_data = {
            'condition': condition_normalized,
            'output': outputs_normalized
        }
        normalized_valid_data.append(normalized_data)

    # Save normalized training and validation data to JSON files
    train_output_file = os.path.join(data_dir, 'train.json')
    valid_output_file = os.path.join(data_dir, 'valid.json')

    with open(train_output_file, 'w') as f:
        json.dump(normalized_train_data, f, indent=4)
    print(f"Training data has been normalized and saved to {train_output_file}")

    with open(valid_output_file, 'w') as f:
        json.dump(normalized_valid_data, f, indent=4)
    print(f"Validation data has been normalized and saved to {valid_output_file}")

    # Save the scalers to disk for future use during model inference or generation
    joblib.dump(scaler_condition, os.path.join(data_dir, 'scaler_condition.pkl'))
    joblib.dump(scaler_output, os.path.join(data_dir, 'scaler_output.pkl'))
    print(f"Scalers have been saved to {data_dir}")


# Example usage:
# This part of the script provides an example of how to use the preprocessing function.

# Load params.json file that contains model settings
params = Params("tests/truecondW1/params.json")

# Define paths to metadata and data directory
metadata_file = "data_raw/data/split_data/file_metadata.json"
data_dir = "data_raw/data/split_data"

# Call the preprocessing function to process the data
preprocess_data(metadata_file, data_dir, params)
