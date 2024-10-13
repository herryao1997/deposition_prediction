# -*- coding: utf-8 -*-
"""
pyqt_inference.py

This script offers the graphic user interface for better interaction during the inference process.

Author: Chengxi Yao
Email: stevenyao@g.skku.edu
"""
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout
import subprocess


class InferenceApp(QWidget):
    def __init__(self):
        """Initialize the application window and set up the UI."""
        super().__init__()

        # Initialize the UI components
        self.initUI()

    def initUI(self):
        """Set up the User Interface components."""
        # Create labels and input fields for the species inputs
        self.species1_label = QLabel('Species 1:', self)
        self.species1_input = QLineEdit(self)

        self.species2_label = QLabel('Species 2:', self)
        self.species2_input = QLineEdit(self)

        self.species3_label = QLabel('Species 3:', self)
        self.species3_input = QLineEdit(self)

        # Create buttons for running inference and exiting the app
        self.run_button = QPushButton('Run Inference', self)
        self.run_button.clicked.connect(self.run_inference)

        self.exit_button = QPushButton('Exit', self)
        self.exit_button.clicked.connect(self.close_application)

        # Set up the layout using horizontal and vertical layouts
        layout = QVBoxLayout()

        # Add input fields to the layout with labels
        h_layout1 = QHBoxLayout()
        h_layout1.addWidget(self.species1_label)
        h_layout1.addWidget(self.species1_input)

        h_layout2 = QHBoxLayout()
        h_layout2.addWidget(self.species2_label)
        h_layout2.addWidget(self.species2_input)

        h_layout3 = QHBoxLayout()
        h_layout3.addWidget(self.species3_label)
        h_layout3.addWidget(self.species3_input)

        layout.addLayout(h_layout1)
        layout.addLayout(h_layout2)
        layout.addLayout(h_layout3)

        # Add buttons to the layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.exit_button)

        layout.addLayout(button_layout)

        # Set the main layout of the window
        self.setLayout(layout)
        self.setWindowTitle('WGAN-GP based Prediction on Deposition Mapping')

    def run_inference(self):
        """Run the inference script based on the input species."""
        # Get the input values for the species
        species1 = self.species1_input.text().strip()
        species2 = self.species2_input.text().strip()
        species3 = self.species3_input.text().strip()

        # Validate input fields
        if not all([species1, species2, species3]):
            print("Error: All species inputs must be filled.")
            return

        # Build the condition string
        condition = f"{species1}-{species2}-{species3}"

        # Run the inference script using subprocess
        try:
            subprocess.run([
                sys.executable, "inference.py",  # Use sys.executable to get the current Python path
                "--params_path", "tests/truecondW1/params.json",
                "--restore_file", "g_best.pth.tar",
                "--location_csv", "location.CSV",
                "--condition", condition,
                "--output_folder", "output",
                "--coordinate_type", "polar",  # Change this to 'cartesian' if needed
                "--data_dir", "data_raw/data/split_data",
                "--test_dir", "tests/truecondW1"
            ], check=True)

            print("Inference completed successfully.")

        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running inference: {e}")

        except FileNotFoundError:
            print("Error: The inference script or one of the required files was not found.")

    def close_application(self):
        """Close the application."""
        self.close()


if __name__ == '__main__':
    # Create the application instance
    app = QApplication(sys.argv)

    # Initialize and display the application window
    ex = InferenceApp()
    ex.show()

    # Execute the application event loop
    sys.exit(app.exec_())

