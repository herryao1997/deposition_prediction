# -*- coding: utf-8 -*-
"""
inference.py

Perform inference using the trained Generator to generate a single image.
The script loads a pretrained model, standardizes the condition input,
and generates a prediction based on a given coordinate system.

Author: Chengxi Yao
Email: stevenyao@g.skku.edu
"""

import argparse
import logging
import os
import numpy as np
import torch
import matplotlib

matplotlib.use("TkAgg")  # Use non-interactive backend
import pandas as pd
import joblib
import utils
import model.cWGAN as cgan

# Argument parsing
# Parsing input arguments for the script
parser = argparse.ArgumentParser()
parser.add_argument("--params_path", default="tests/truecondW1/params.json",
                    help="Path to the params.json file containing model parameters")
parser.add_argument("--restore_file", default="g_best.pth.tar",
                    help="File containing the trained Generator model weights")
parser.add_argument("--location_csv", default="location.csv",
                    help="CSV file containing the coordinates of points")
parser.add_argument("--condition", default="1.0-2.0-3.0",
                    help="Condition vector as a string, e.g., '1.0-2.0-3.0'")
parser.add_argument("--output_folder", default="output",
                    help="Folder to save the generated image")
parser.add_argument("--coordinate_type", default="cartesian",
                    help="Type of coordinate system: 'cartesian' or 'polar'")
parser.add_argument("--data_dir", default="data_raw/data/split_data",
                    help="Directory containing the saved scalers")
parser.add_argument("--test_dir", default="tests/truecondW1",
                    help="Directory containing the trained model")


def inference(model: cgan.Generator, params: utils.Params, condition: list,
              location_csv: str, output_folder: str, scaler_condition, scaler_output,
              coordinate_type: str) -> None:
    """
    Perform inference using the trained generator.

    @param model Trained generator model.
    @param params Model hyperparameters.
    @param condition Input condition vector as a list of floats.
    @param location_csv Path to the CSV file containing coordinates.
    @param output_folder Folder to save the generated images.
    @param scaler_condition Scaler used to normalize the condition vector.
    @param scaler_output Scaler used to inverse transform the output vector.
    @param coordinate_type Coordinate system type, either 'cartesian' or 'polar'.

    @return None
    """
    model.eval()

    # Load coordinate data
    coords1, coords2 = utils.load_coordinates(location_csv, coordinate_type)

    # Normalize condition vector
    condition_normalized = scaler_condition.transform([condition])

    # Generate random noise vector
    noise = torch.randn(1, params.z_dim)

    # Convert condition to tensor
    conditions_tensor = torch.tensor(condition_normalized).float()

    if params.cuda:
        noise, conditions_tensor = noise.cuda(), conditions_tensor.cuda()
        model = model.cuda()

    # Generate thickness values
    with torch.no_grad():
        thickness_normalized = model(noise, conditions_tensor).cpu().numpy()

    # Inverse transform thickness data
    thickness = scaler_output.inverse_transform(thickness_normalized)

    # Flatten thickness array
    thickness = thickness.flatten()

    # Verify lengths match
    assert len(coords1) == len(thickness), "Mismatch between number of coordinates and thickness values"

    # Plot and save images
    img_name = "inferred_image"
    utils.process_and_plot(coords1, coords2, thickness, img_name, output_folder, coordinate_type)


# Main block of the script
if __name__ == "__main__":
    # Parse the arguments provided by the user
    args = parser.parse_args()

    # Load the parameters from the params.json file
    params = utils.Params(args.params_path)
    params.cuda = torch.cuda.is_available()

    # Set random seed for reproducibility
    torch.manual_seed(340)
    if params.cuda:
        torch.cuda.manual_seed(340)

    # Set up logging to record the inference process
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    utils.set_logger(os.path.join(args.output_folder, "inference.log"))

    logging.info("Loading trained model and scalers...")

    # Load the scalers used for normalizing condition inputs and inverse transforming outputs
    scaler_condition = joblib.load(os.path.join(args.data_dir, 'scaler_condition.pkl'))
    scaler_output = joblib.load(os.path.join(args.data_dir, 'scaler_output.pkl'))

    # Initialize the generator model
    model = cgan.Generator(params)

    # Load the trained generator weights from the specified checkpoint
    checkpoint_dir = os.path.join(args.test_dir, 'checkpoints')
    model_path = os.path.join(checkpoint_dir, args.restore_file)
    utils.load_checkpoint(model_path, model)

    logging.info("Starting inference...")

    # Parse the condition vector from the input string
    condition = [float(i) for i in args.condition.split('-')]

    # Perform inference and generate the image
    inference(model, params, condition, args.location_csv, args.output_folder,
              scaler_condition, scaler_output, args.coordinate_type)

    logging.info("Inference completed.")
