# -*- coding: utf-8 -*-
"""
utils.py

Contains utility functions and classes for the project, including
file handling, logging, hyperparameter management, training utilities,
metric computations, and data visualization.

Author: Chengxi Yao
Email: stevenyao@g.skku.edu
"""

import os
import shutil
import json
import logging
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm  # <-- make sure the cm color map has been loaded
from scipy.interpolate import griddata, LinearNDInterpolator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import torch
import seaborn as sns

# ---------- File Handling and Logging ---------- #

def load_scaler(scaler_path: str):
    """
    Load a pre-trained scaler using joblib.

    @param scaler_path: Path to the scaler file.
    @return: Loaded scaler.
    """
    return joblib.load(scaler_path)


def set_logger(log_path: str) -> None:
    """
    Set up the logger to log messages to both console and file.

    @param log_path: Path to the log file.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s:%(levelname)s: %(message)s"))
    logger.addHandler(file_handler)

    # Console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)


def save_dict_to_json(d: dict, json_path: str) -> None:
    """
    Save a dictionary of floats to a JSON file.

    @param d: Dictionary to save.
    @param json_path: Path to the JSON file.
    """
    with open(json_path, "w") as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state: dict, is_best: bool, checkpoint_dir: str,
                    model: Optional[str] = None) -> None:
    """
    Save the model and training parameters.

    @param state: Model's state dictionary.
    @param is_best: True if it is the best model seen till now.
    @param checkpoint_dir: Directory to save the checkpoint.
    @param model: 'gen' or 'disc' indicating the model type.
    """
    # Ensure the checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if model == "gen":
        last = "g_last.pth.tar"
        best = "g_best.pth.tar"
    elif model == "disc":
        last = "d_last.pth.tar"
        best = "d_best.pth.tar"
    else:
        last = "last.pth.tar"
        best = "best.pth.tar"

    filepath = os.path.join(checkpoint_dir, last)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, best))


def load_checkpoint(checkpoint: str, model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None):
    """
    Load model parameters (state_dict) from file_path. If optimizer is provided,
    loads state_dict of optimizer assuming it is present in checkpoint.

    @param checkpoint: Filename which needs to be loaded.
    @param model: Model for which the parameters are loaded.
    @param optimizer: Resume optimizer from checkpoint.
    """
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"File doesn't exist {checkpoint}")
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optim_dict"])

    return checkpoint


# ---------- Hyperparameter Management ---------- #

class Params:
    """
    Class that loads hyperparameters from a JSON file.
    """

    def __init__(self, json_path: str):
        """
        Initialize with hyperparameters from the JSON file.

        @param json_path: Path to the JSON file.
        """
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path: str):
        """
        Save parameters to a JSON file.

        @param json_path: Path to save the parameters.
        """
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path: str):
        """
        Update parameters from a JSON file.

        @param json_path: Path to the JSON file.
        """
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """
        Access parameters as a dictionary.

        @return: Dictionary of parameters.
        """
        return self.__dict__


# ---------- Training Utilities ---------- #

class RunningAverage:
    """
    A class that maintains the running average of a quantity.
    """

    def __init__(self) -> None:
        self.steps = 0
        self.total = 0

    def update(self, val: float) -> None:
        """
        Update the running average with a new value.

        @param val: New value to update the average.
        """
        self.total += val
        self.steps += 1

    def __call__(self) -> float:
        """
        Compute the current running average.

        @return: Current average.
        """
        return self.total / float(self.steps)


# ---------- Metric Functions ---------- #

def compute_mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Compute Mean Squared Error between true and predicted values.

    @param y_true: True values.
    @param y_pred: Predicted values.
    @return: MSE value.
    """
    mse_loss = torch.nn.functional.mse_loss(y_pred, y_true, reduction='mean')
    return mse_loss.item()


def compute_pcc(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Compute Pearson Correlation Coefficient between true and predicted values.

    @param y_true: True values.
    @param y_pred: Predicted values.
    @return: PCC value.
    """
    y_true_mean = torch.mean(y_true, dim=1, keepdim=True)
    y_pred_mean = torch.mean(y_pred, dim=1, keepdim=True)

    cov = torch.mean((y_true - y_true_mean) * (y_pred - y_pred_mean), dim=1)
    y_true_std = torch.std(y_true, dim=1)
    y_pred_std = torch.std(y_pred, dim=1)

    pcc = cov / (y_true_std * y_pred_std + 1e-8)
    return torch.mean(pcc).item()


# ---------- Plotting Functions ---------- #

def plot_metrics(epochs, metrics, labels, ylabel, title, filename, test_dir):
    """
    Plot and save metrics curves.

    @param epochs: List of epoch numbers.
    @param metrics: List containing metric values for each curve.
    @param labels: List of labels for the curves.
    @param ylabel: Label for the y-axis.
    @param title: Title of the plot.
    @param filename: Filename to save the plot.
    @param test_dir: Directory to save the plot.
    """
    plt.figure(figsize=(8, 6))
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        if i == 0:
            plt.plot(epochs, metric, '-o', color='red', label=label)  # First metric with red solid line
        elif i == 1:
            plt.plot(epochs, metric, '-^', color='blue', label=label)  # Second metric with green dashed line
        else:
            plt.plot(epochs, metric, '-*', color='green', label=label)  # Third metric with blue markers

    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.title(title, fontsize=16)
    plt.tick_params(axis='both', labelsize=13)
    plt.grid(True)
    plt.legend(loc='best', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(test_dir, filename), dpi=500)
    plt.close()
    logging.info(f"{title} plot saved to {os.path.join(test_dir, filename)}")



def plot_loss_curve(epochs, g_losses, d_losses, val_g_losses, test_dir):
    """
    Plot loss curves and save to the specified directory.

    @param epochs: List of epoch numbers.
    @param g_losses: Generator loss values.
    @param d_losses: Discriminator loss values.
    @param val_g_losses: Validation generator loss values.
    @param test_dir: Directory to save the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, g_losses, '-o',  color='red', label='Generator Loss')
    plt.plot(epochs, d_losses, '-^', color='blue', label='Discriminator Loss')
    plt.plot(epochs, val_g_losses, '-*', color='green', label='Validation Generator Loss')

    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.tick_params(axis='both', labelsize=13)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize=13)
    plt.tight_layout()

    plt.savefig(os.path.join(test_dir, 'loss_curve.png'), dpi=500)
    plt.close()
    logging.info(f"Loss curves have been saved to {os.path.join(test_dir, 'loss_curve.png')}")


# ---------- Data Visualization ---------- #

def load_coordinates(location_csv: str, coordinate_type: str):
    """
    Load coordinate data from a CSV file and return relevant coordinate arrays.

    @param location_csv: Path to the CSV file containing location data.
    @param coordinate_type: Type of coordinates ('cartesian' or 'polar').
    @return: Arrays containing the x, y, or r, theta coordinates.
    """
    location_data = pd.read_csv(location_csv)
    if coordinate_type == 'cartesian':
        x_coords = location_data['x(mm)'].values
        y_coords = location_data['y(mm)'].values
        valid_indices = ~np.isnan(x_coords) & ~np.isnan(y_coords)
        x_coords = x_coords[valid_indices]
        y_coords = y_coords[valid_indices]
        return x_coords, y_coords
    elif coordinate_type == 'polar':
        r_coords = location_data['R(mm)'].values
        theta_coords_deg = location_data['theta(deg)'].values
        valid_indices = ~np.isnan(r_coords) & ~np.isnan(theta_coords_deg)
        r_coords = r_coords[valid_indices]
        theta_coords_deg = theta_coords_deg[valid_indices]
        theta_coords_rad = np.deg2rad(theta_coords_deg)
        return r_coords, theta_coords_rad
    else:
        raise ValueError(f"Unknown coordinate_type {coordinate_type}, choose 'cartesian' or 'polar'.")


def process_and_plot(coords1, coords2, thickness, img_name, output_folder, coordinate_type):
    """
    Process thickness data, plot both 2D and 3D plots based on the selected coordinate system, and display them side by side.

    @param coords1: x or r coordinates.
    @param coords2: y or theta coordinates.
    @param thickness: Thickness values.
    @param img_name: Base name of the saved images (without extension).
    @param output_folder: Folder to save the images.
    @param coordinate_type: Coordinate system type ('cartesian' or 'polar').
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if coordinate_type == 'cartesian':
        img_path_2d = plot_cartesian(coords1, coords2, thickness, img_name + '_2d.png', output_folder)
        img_path_3d = plot_3d_cartesian(coords1, coords2, thickness, img_name + '_3d.png', output_folder)
    elif coordinate_type == 'polar':
        img_path_2d = plot_polar(coords1, coords2, thickness, img_name + '_2d.png', output_folder)
        img_path_3d = plot_3d_polar(coords1, coords2, thickness, img_name + '_3d.png', output_folder)
    else:
        raise ValueError(f"Unknown coordinate_type '{coordinate_type}', choose 'cartesian' or 'polar'.")

    img_2d = plt.imread(img_path_2d)
    img_3d = plt.imread(img_path_3d)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    axes[0].imshow(img_2d)
    axes[0].axis('off')
    axes[0].set_title('2D Plot', fontsize=12)

    pos1 = axes[0].get_position()
    axes[0].set_position([pos1.x0, pos1.y0 + 0.1, pos1.width * 0.8, pos1.height * 0.8])

    axes[1].imshow(img_3d)
    axes[1].axis('off')
    axes[1].set_title('3D Plot', fontsize=12)

    combined_img_path = os.path.join(output_folder, img_name + '_combined.png')
    plt.tight_layout(pad=2.0)
    plt.savefig(combined_img_path, dpi=300)

    plt.show()

    logging.info(f"Combined image saved to {combined_img_path}")


def plot_cartesian(x, y, thickness, img_name, output_folder):
    """
    Plot and save a Cartesian coordinate heatmap with conditional extrapolation.

    @param x: x coordinates.
    @param y: y coordinates.
    @param thickness: Thickness values.
    @param img_name: Name of the saved image.
    @param output_folder: Folder to save the image.
    @return: Path to the saved image.
    """
    # Create grid over the range of data
    grid_x, grid_y = np.mgrid[min(x):max(x):200j, min(y):max(y):200j]
    points = np.array([x, y]).T

    # Initial interpolation without extrapolation
    grid_values = griddata(points, thickness, (grid_x, grid_y), method='cubic')

    # Check for NaN values in the interpolated grid
    nan_count = np.isnan(grid_values).sum()
    print(f"Initial interpolation has {nan_count} NaN values out of {grid_values.size} total values.")

    # If NaNs are present, perform extrapolation
    if nan_count > 0:
        print("Performing extrapolation to fill missing values.")
        # Use LinearNDInterpolator for extrapolation
        interpolator = LinearNDInterpolator(points, thickness)
        grid_values = interpolator(grid_x, grid_y)
        # Fill any remaining NaNs with average thickness
        grid_values = np.nan_to_num(grid_values, nan=np.nanmean(thickness))

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.contourf(grid_x, grid_y, grid_values, 100, cmap='viridis')
    cbar = plt.colorbar()
    cbar.set_label('Thickness (nm)', fontsize=12)
    cbar.set_clim(320, 900)  # this is for the colormap uniform clipping.

    # Scatter original data points
    plt.scatter(x, y, c='black', s=20, label='Data Points')

    # Set labels and title
    plt.xlabel('X (mm)', fontsize=15)
    plt.ylabel('Y (mm)', fontsize=15)
    plt.tick_params(axis='both', labelsize=13)
    plt.grid(True)
    plt.legend(loc='upper center')

    # Save and close the plot
    plt.tight_layout()
    img_path = os.path.join(output_folder, img_name)
    plt.savefig(img_path, dpi=300)
    plt.close()

    return img_path


def plot_polar(r, theta, thickness, img_name, output_folder):
    """
    Plot and save a polar coordinate heatmap, ensuring continuity at 360 degrees.

    @param r: Radial coordinates.
    @param theta: Angular coordinates in radians.
    @param thickness: Thickness values.
    @param img_name: Name of the saved image.
    @param output_folder: Folder to save the image.
    @return: Path to the saved image.
    """
    theta = np.array(theta)
    r = np.array(r)
    thickness = np.array(thickness)

    # Wrap data at theta=2π by adding data point(s) at theta=2π equivalent to theta=0
    zero_theta_indices = np.isclose(theta, 0) | np.isclose(theta, 2*np.pi)
    if np.any(zero_theta_indices):
        # Get corresponding r and thickness values at theta=0
        r_zero = r[zero_theta_indices]
        thickness_zero = thickness[zero_theta_indices]
        # Append data at theta=2π
        r = np.concatenate((r, r_zero))
        theta = np.concatenate((theta, np.full_like(r_zero, 2*np.pi)))
        thickness = np.concatenate((thickness, thickness_zero))
    else:
        # If no data at theta=0, add first point at theta=2π
        r = np.concatenate((r, [r[0]]))
        theta = np.concatenate((theta, [2*np.pi]))
        thickness = np.concatenate((thickness, [thickness[0]]))

    # Create grid over full 0 to 2π range
    grid_r, grid_theta = np.mgrid[min(r):max(r):200j, 0:2*np.pi:200j]
    points = np.array([r, theta]).T

    # Interpolation
    grid_values = griddata(points, thickness, (grid_r, grid_theta), method='cubic')

    # Handle NaN values
    nan_count = np.isnan(grid_values).sum()
    if nan_count > 0:
        # Use LinearNDInterpolator with extrapolation
        interpolator = LinearNDInterpolator(points, thickness)
        grid_values = interpolator(grid_r, grid_theta)
        # Fill any remaining NaNs with the mean thickness
        grid_values = np.nan_to_num(grid_values, nan=np.nanmean(thickness))

    # Plotting
    plt.figure(figsize=(8, 6))
    ax = plt.subplot(111, projection='polar')

    # Plot the interpolated grid
    c = ax.pcolormesh(grid_theta, grid_r, grid_values, cmap='viridis', shading='auto')

    # Add color bar with label
    cbar = plt.colorbar(c, ax=ax)
    cbar.set_label('Thickness (nm)', fontsize=12)
    c.set_clim(320, 900)  # this is for the colormap uniform clipping.

    # Scatter original data points
    ax.scatter(theta, r, c='red', s=20, label='Data Points')

    # Set plot limits and labels
    ax.set_ylim([min(r), max(r)])
    ax.tick_params(axis='both', labelsize=13)
    ax.grid(True)
    ax.legend(loc='upper center')

    # Save and close the plot
    plt.tight_layout()
    img_path = os.path.join(output_folder, img_name)
    plt.savefig(img_path, dpi=300)
    plt.close()

    return img_path


def plot_3d_polar(r, theta, thickness, img_name, output_folder):
    """
    Directly compute and plot a 3D surface in polar coordinates, handling periodicity (0 degrees = 360 degrees).

    @param r: Radial coordinates.
    @param theta: Angular coordinates (in radians).
    @param thickness: Thickness values.
    @param img_name: Name of the saved image (e.g., '3d_polar.png').
    @param output_folder: Folder to save the image.
    @return: Path to the saved image.
    """
    r = np.array(r)
    theta = np.array(theta)
    thickness = np.array(thickness)

    # Handle periodicity (0 degrees = 360 degrees)
    theta = theta % (2 * np.pi)

    # Check if data points at theta = 2π need to be added to ensure angular continuity
    if not np.isclose(theta, 2 * np.pi).any():
        zero_theta_indices = np.isclose(theta, 0)
        if np.any(zero_theta_indices):
            # Add data points at theta = 2π
            r_extra = r[zero_theta_indices]
            thickness_extra = thickness[zero_theta_indices]
            theta_extra = np.full_like(r_extra, 2 * np.pi)
            r = np.concatenate([r, r_extra])
            theta = np.concatenate([theta, theta_extra])
            thickness = np.concatenate([thickness, thickness_extra])
    else:
        # If data at theta = 2π already exists, ensure data at theta = 0 also exists
        two_pi_theta_indices = np.isclose(theta, 2 * np.pi)
        if not np.isclose(theta, 0).any():
            r_extra = r[two_pi_theta_indices]
            thickness_extra = thickness[two_pi_theta_indices]
            theta_extra = np.full_like(r_extra, 0)
            r = np.concatenate([r, r_extra])
            theta = np.concatenate([theta, theta_extra])
            thickness = np.concatenate([thickness, thickness_extra])

    # Create polar coordinate grid
    r_min, r_max = np.min(r), np.max(r)
    r_lin = np.linspace(r_min, r_max, 200)
    theta_lin = np.linspace(0, 2 * np.pi, 200)
    Theta, R = np.meshgrid(theta_lin, r_lin)

    # Interpolate thickness values on the polar grid
    points = np.array([theta, r]).T
    grid_z = griddata(points, thickness, (Theta, R), method='cubic')

    # Handle NaN values (mask NaN values)
    grid_z_masked = np.ma.array(grid_z, mask=np.isnan(grid_z))

    # Convert polar grid to Cartesian coordinates for plotting
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    # Create figure and axes
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface, removing grid lines
    surf = ax.plot_surface(X, Y, grid_z_masked, cmap=cm.viridis,
                           linewidth=0, antialiased=True, alpha=0.8, edgecolor='none')
    # Add projections on three planes, using ax.contour instead of ax.contourf
    ax.contourf(X, Y, grid_z, zdir='z', offset=np.nanmin(grid_z), cmap=cm.viridis)

    # Add original data points
    x_data = r * np.cos(theta)
    y_data = r * np.sin(theta)
    ax.scatter(x_data, y_data, thickness, c='red', marker='o', s=50, label='Data Points')
    ax.legend()
    ax.grid(False)
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    ax.tick_params(axis='x', which='both', direction='out')
    ax.tick_params(axis='y', which='both', direction='out')
    ax.tick_params(axis='z', which='both', direction='out')

    # Set labels and limits
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Thickness (nm)')
    ax.set_xlim(np.min(X), np.max(X))
    ax.set_ylim(np.min(Y), np.max(Y))
    ax.set_zlim(np.nanmin(grid_z), np.nanmax(grid_z))

    # Add colorbar
    cbar = fig.colorbar(surf, shrink=0.5, aspect=10, label='Thickness (nm)')
    surf.set_clim(320, 900)

    # Adjust view angle (optional)
    ax.view_init(elev=30, azim=45)

    # Save and close the plot
    img_path = os.path.join(output_folder, img_name)
    plt.tight_layout()
    plt.savefig(img_path, dpi=300)
    plt.close()

    return img_path


def plot_3d_cartesian(x, y, thickness, img_name, output_folder):
    """
    Plot and save a 3D surface plot in Cartesian coordinates, handling NaN values in interpolation.

    @param x: x coordinates.
    @param y: y coordinates.
    @param thickness: Thickness values.
    @param img_name: Name of the saved image (e.g., '3d_cartesian.png').
    @param output_folder: Folder to save the image.
    @return: Path to the saved image.
    """
    # Create a grid
    grid_x, grid_y = np.mgrid[min(x):max(x):200j, min(y):max(y):200j]

    # Initial interpolation (without extrapolation)
    grid_z = griddata((x, y), thickness, (grid_x, grid_y), method='cubic')

    # Check for NaN values in the interpolated grid
    nan_count = np.isnan(grid_z).sum()
    print(f"grid_z has {nan_count} NaN values out of {grid_z.size} total values.")

    # If NaN values exist, perform extrapolation
    if nan_count > 0:
        print("Performing extrapolation to fill missing values in grid_z.")
        interpolator = LinearNDInterpolator(np.column_stack((x, y)), thickness)
        grid_z = interpolator(grid_x, grid_y)
        # Fill remaining NaN values
        nan_count_after = np.isnan(grid_z).sum()
        if nan_count_after > 0:
            print(f"Filling remaining {nan_count_after} NaN values with mean thickness.")
            grid_z = np.nan_to_num(grid_z, nan=np.nanmean(thickness))

    # Ensure there are no NaN values in grid_z
    if np.isnan(grid_z).any():
        raise ValueError("grid_z still contains NaN values after extrapolation and filling.")

    # Create figure and axes
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap=cm.viridis,
                           linewidth=0, antialiased=True, alpha=0.8, edgecolor='none')

    # Add projected contours
    ax.contourf(grid_x, grid_y, grid_z, zdir='z', offset=np.min(grid_z), cmap=cm.viridis, alpha=0.5)
    ax.contourf(grid_x, grid_y, grid_z, zdir='x', offset=np.min(grid_x), cmap=cm.viridis, alpha=0.5)
    ax.contourf(grid_x, grid_y, grid_z, zdir='y', offset=np.max(grid_y), cmap=cm.viridis, alpha=0.5)

    # Set labels and limits
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Thickness (nm)')
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    ax.set_zlim(np.min(grid_z), np.max(grid_z))

    # Add colorbar
    cbar = fig.colorbar(surf, shrink=0.5, aspect=10, label='Thickness (nm)')
    surf.set_clim(320, 900)

    # Save and close the plot
    img_path = os.path.join(output_folder, img_name)
    plt.tight_layout()
    plt.savefig(img_path, dpi=300)
    plt.close()

    return img_path
