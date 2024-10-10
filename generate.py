# -*- coding: utf-8 -*-
"""
generate.py

Generate multiple images using a trained Generator model based on the dataset.

Author: Chengxi Yao
Email: stevenyao@g.skku.edu
"""

import argparse
import logging
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import joblib
import utils
import model.cWGAN as cgan
from model.dataloader import DepositDataset as DD

# Argument parsing
# Parsing the input arguments for the script
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data_raw/data/split_data",
                    help="Directory containing the dataset and scalers")
parser.add_argument("--test_dir", default="tests/truecondW1",
                    help="Directory containing params.json and the trained model")
parser.add_argument("--restore_file", default="g_best.pth.tar",
                    help="File containing trained generator model weights")
parser.add_argument("--location_csv", default="location.csv",
                    help="Path to the CSV file containing coordinates")
parser.add_argument("--coordinate_type", default="cartesian",
                    help="Coordinate system type ('cartesian' or 'polar')")


def generate_images(model: cgan.Generator, params: utils.Params,
                    dataloader: DataLoader, output_folder: str, location_csv: str,
                    scaler_condition, scaler_output, coordinate_type: str) -> None:
    """
    Generate images using the trained generator model.

    This function uses the generator model to generate multiple images based on
    the input dataset and saves them to the specified output folder.

    @param model Trained generator model.
    @param params Model hyperparameters.
    @param dataloader DataLoader for the dataset.
    @param output_folder Folder to save the generated images.
    @param location_csv Path to the CSV file containing coordinates.
    @param scaler_condition Scaler for normalizing condition.
    @param scaler_output Scaler for inverse transforming output.
    @param coordinate_type Coordinate system type ('cartesian' or 'polar').

    @return None
    """
    model.eval()  # Set model to evaluation mode

    # Load coordinate data from CSV file
    coords1, coords2 = utils.load_coordinates(location_csv, coordinate_type)

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through the dataset and generate images
    with tqdm(total=len(dataloader)) as t:
        for i, (conditions, _) in enumerate(dataloader):
            # Standardize condition input using the preloaded scaler
            condition_normalized = scaler_condition.transform([conditions[0].tolist()])
            noise = torch.randn(1, params.z_dim)  # Generate random noise
            conditions_tensor = torch.tensor(condition_normalized).float()

            # Move tensors to GPU if available
            if params.cuda:
                noise, conditions_tensor = noise.cuda(), conditions_tensor.cuda()

            # Generate thickness values using the generator model
            with torch.no_grad():
                thickness_normalized = model(noise, conditions_tensor).cpu().numpy()

            # Inverse transform the generated output to its original scale
            thickness = scaler_output.inverse_transform(thickness_normalized)

            # Plot and save the generated image
            img_name = f"generated_image_{i}.jpg"
            utils.process_and_plot(coords1, coords2, thickness[0], img_name, output_folder, coordinate_type)

            t.update()

    logging.info(f"Generated images saved to {output_folder}")


if __name__ == "__main__":
    # Parse arguments provided by the user
    args = parser.parse_args()

    # Load params.json file that contains the model hyperparameters
    json_path = os.path.join(args.test_dir, "params.json")
    assert os.path.isfile(json_path), f"No json configuration file found at {json_path}"
    params = utils.Params(json_path)

    # Check if CUDA (GPU) is available
    params.cuda = torch.cuda.is_available()

    # Set random seed for reproducibility
    torch.manual_seed(340)
    if params.cuda:
        torch.cuda.manual_seed(340)

    # Set up logging to record the generation process
    output_folder = os.path.join(args.test_dir, 'generated_images')
    utils.set_logger(os.path.join(output_folder, "generation.log"))

    logging.info("Loading the dataset and scalers...")

    # Load scalers for normalizing and inverse transforming conditions and outputs
    scaler_condition = joblib.load(os.path.join(args.data_dir, 'scaler_condition.pkl'))
    scaler_output = joblib.load(os.path.join(args.data_dir, 'scaler_output.pkl'))

    # Set up the dataset and DataLoader
    dataset = DD(os.path.join(args.data_dir, 'train.json'))
    dl = DataLoader(dataset, batch_size=1, shuffle=True,
                    num_workers=params.num_workers, pin_memory=params.cuda)

    logging.info("- Dataset and scalers loaded successfully.")

    # Load the trained generator model from the checkpoint
    model = cgan.Generator(params).cuda() if params.cuda else cgan.Generator(params)

    logging.info("Starting generation...")

    # Load trained generator weights from the specified file
    checkpoint_dir = os.path.join(args.test_dir, 'checkpoints')
    model_path = os.path.join(checkpoint_dir, args.restore_file)
    utils.load_checkpoint(model_path, model)

    # Call the function to generate images
    generate_images(model, params, dl, output_folder, args.location_csv,
                    scaler_condition, scaler_output, args.coordinate_type)

    logging.info("Generation completed.")
