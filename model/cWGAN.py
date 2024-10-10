# -*- coding: utf-8 -*-
"""
cWGAN.py

Contains the generator and discriminator models and the metrics used
to evaluate them. This file defines the architectures for a conditional
Wasserstein GAN (cWGAN) with Gradient Penalty (WGAN-GP), including
custom evaluation metrics like MSE, variance, and a main metric combining the two.

Author: Chengxi Yao
Email: stevenyao@g.skku.edu
"""

import sys
sys.path.append("../")

import torch
import torch.nn as nn
from utils import Params


class Generator(nn.Module):
    """
    Generator model for conditional WGAN-GP.

    This model takes in noise (z) and a conditional input (y), and generates
    synthetic data that mimics the real data distribution.

    @param params Model hyperparameters including activation function,
                  input dimensions, number of nodes, and output dimensions.
    """
    def __init__(self, params: Params) -> None:
        """
        Constructor for the Generator class.

        @param params Hyperparameters for the Generator including activation function,
                      input dimensions, and number of nodes.
        """
        super(Generator, self).__init__()

        # Set activation function and parameters
        if params.gen_act_func == "relu":
            self.act_func = nn.ReLU
            self.func_params = {"inplace": True}
        elif params.gen_act_func == "leaky_relu":
            self.act_func = nn.LeakyReLU
            self.func_params = {"negative_slope": 0.2, "inplace": True}
        elif params.gen_act_func == "elu":
            self.act_func = nn.ELU
            self.func_params = {"alpha": 1.0, "inplace": True}
        else:
            raise ValueError(f"Unsupported activation function: {params.gen_act_func}")

        # Define noise processing layers
        self.noise = nn.Sequential(
            nn.Linear(params.z_dim, params.num_nodes),
            self.act_func(**self.func_params),
            nn.Linear(params.num_nodes, params.num_nodes // 2),
            self.act_func(**self.func_params),
        )

        # Define condition processing layers
        self.cond = nn.Sequential(
            nn.Linear(params.input_dim, params.num_nodes),
            self.act_func(**self.func_params),
            nn.Linear(params.num_nodes, params.num_nodes // 2),
            self.act_func(**self.func_params),
        )

        # Define output layers
        self.out = nn.Sequential(
            nn.Linear(params.num_nodes, params.num_nodes),
            self.act_func(**self.func_params),
            nn.Linear(params.num_nodes, params.num_nodes),
            self.act_func(**self.func_params),
            nn.Linear(params.num_nodes, params.output_dim),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Generator.

        Combines noise (z) and conditional input (y) to generate synthetic data.

        @param z Noise vector.
        @param y Conditional input vector.

        @return Generated data as a torch.Tensor.
        """
        z = self.noise(z)
        y = self.cond(y)
        combined = torch.cat([z, y], dim=-1)
        x = self.out(combined)
        return x


class Discriminator(nn.Module):
    """
    Discriminator model for conditional WGAN-GP.

    This model evaluates the authenticity of the generated (or real) data
    by processing both the data and the conditional input.

    @param params Model hyperparameters including activation function,
                  input dimensions, number of nodes, and output dimensions.
    """
    def __init__(self, params: Params):
        """
        Constructor for the Discriminator class.

        @param params Hyperparameters for the Discriminator including activation function,
                      input dimensions, and number of nodes.
        """
        super(Discriminator, self).__init__()

        # Set activation function and parameters
        if params.disc_act_func == "relu":
            self.act_func = nn.ReLU
            self.func_params = {"inplace": True}
        elif params.disc_act_func == "leaky_relu":
            self.act_func = nn.LeakyReLU
            self.func_params = {"negative_slope": 0.2, "inplace": True}
        elif params.disc_act_func == "elu":
            self.act_func = nn.ELU
            self.func_params = {"alpha": 1.0, "inplace": True}
        else:
            raise ValueError(f"Unsupported activation function: {params.disc_act_func}")

        # Define generator output processing layers
        self.genout = nn.Sequential(
            nn.Linear(params.output_dim, params.num_nodes),
            self.act_func(**self.func_params),
            nn.Dropout(p=0.3),
            nn.Linear(params.num_nodes, params.num_nodes // 2),
            self.act_func(**self.func_params),
            nn.Dropout(p=0.3),
        )

        # Define condition processing layers
        self.cond = nn.Sequential(
            nn.Linear(params.input_dim, params.num_nodes),
            self.act_func(**self.func_params),
            nn.Dropout(p=0.3),
            nn.Linear(params.num_nodes, params.num_nodes // 2),
            self.act_func(**self.func_params),
            nn.Dropout(p=0.3),
        )

        # Define output layers
        self.out = nn.Sequential(
            nn.Linear(params.num_nodes, params.num_nodes),
            self.act_func(**self.func_params),
            nn.Dropout(p=0.3),
            nn.Linear(params.num_nodes, params.num_nodes),
            self.act_func(**self.func_params),
            nn.Dropout(p=0.3),
            nn.Linear(params.num_nodes, 1),  # Output score
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Discriminator.

        Evaluates whether the given data (real or generated) is authentic
        based on the input data (x) and the condition (y).

        @param x Input data (real or generated).
        @param y Conditional input vector.

        @return Discriminator score (real or fake) as a torch.Tensor.
        """
        x = self.genout(x)
        y = self.cond(y)
        combined = torch.cat([x, y], dim=-1)
        out = self.out(combined)
        return out


# ---------- Metrics for Model Evaluation ---------- #

def MSE(out: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """
    Mean Squared Error (MSE) loss function.

    @param out Model output (generated data).
    @param truth Ground truth (real data).

    @return Computed MSE loss as a torch.Tensor.
    """
    mse = nn.MSELoss()
    return mse(out, truth)


def VAR(out: torch.Tensor) -> torch.Tensor:
    """
    Variance of the output data.

    This metric calculates the variance of the model's output, which is
    useful for evaluating the spread or diversity of the generated data.

    @param out Model output (generated data).

    @return Computed variance of the output as a torch.Tensor.
    """
    return torch.var(out)


def main_metric(out: torch.Tensor, truth: torch.Tensor, N: int) -> torch.Tensor:
    """
    Custom metric combining Mean Squared Error (MSE) and Variance (VAR).

    The metric is a weighted combination of MSE and variance, with the
    weight determined by N.

    @param out Model output (generated data).
    @param truth Ground truth (real data).
    @param N Weighting factor.

    @return Computed main metric combining MSE and VAR as a torch.Tensor.
    """
    return N * MSE(out, truth) / (N + 1) + VAR(out) / (N + 1)


# Dictionary of metrics for easy access during model evaluation
metrics: dict = {
    "MSE": MSE,
    "VAR": VAR,
    "main": main_metric
}
