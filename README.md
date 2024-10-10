# Project Workflow

This document outlines the step-by-step process for running the full project workflow from data preparation to model training, inference, and batch generation.

## 1. Data Preparation and Processing:

- **Step 1:** Use `config_csv_json.py` to generate JSON datasets from raw CSV files.

    ```bash
    python data_raw/config_csv_json.py --input_dir data_raw --output_file data_prepared/data.json
    ```

- **Step 2 (Optional):** Use `preprocess.py` to normalize the data (both condition and output vectors).

    ```bash
    python preprocess.py --input_file data_prepared/data.json --output_file data_prepared/normalized_data.json
    ```

## 2. Dataset Preparation:

- **Step 3:** The `dataloader.py` automatically loads the dataset during training. You don't need to run this script separately; it is called by the training script `train_and_evaluate.py`.

## 3. Model Training:

- **Step 4:** Use `train_and_evaluate.py` to train the Generator and Discriminator models, and save the model weights.

    ```bash
    python train_and_evaluate.py --data_dir data_prepared/normalized_data.json --model_dir checkpoints --params_file tests/truecondW1/params.json
    ```

## 4. Inference:

- **Step 5:** Use `inference.py` to perform inference on a single input sample, generate a heatmap, and save the result.

    ```bash
    python inference.py --params_path tests/truecondW1/params.json --restore_file checkpoints/g_best.pth.tar --location_csv data_raw/location.csv --condition "1-2-3" --output_folder output
    ```

## 5. Batch Generation:

- **Step 6:** Use `generate.py` to perform batch inference on multiple test samples and save the generated results.

    ```bash
    python generate.py --data_dir data_prepared/normalized_data.json --test_dir tests/truecondW1 --restore_file checkpoints/g_best.pth.tar --location_csv data_raw/location.csv
    ```

## File Structure:

The directory structure is as follows:

```bash
├── checkpoints
├── data_prep.md
├── data_raw
│   ├── config_csv_json.py
│   ├── json_plot.py
│   ├── plot_deposition.py
│   └── RESULT.CSV
├── generate.py
├── inference.py
├── logs
├── model
│   ├── cWGAN.py
│   ├── dataloader.py
│   └── __pycache__
│       └── dataloader.cpython-38.pyc
├── preprocess.py
├── README.md
├── tests
│   └── truecondW1
│       └── params.json
├── train_and_evaluate.py
└── utils.py
