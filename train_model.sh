#!/bin/bash

# Activate the Python environment (if using virtualenv or conda)
# Replace 'your_env_name' with the name of your environment
# source /path/to/your/virtualenv/bin/activate
# Or for Conda:
# source activate your_env_name

# Define directories and files
TEST_DIR="tests/truecondW1"  # The directory where params.json and the model weights will be saved
DATA_DIR="data_raw/data/split_data"  # The directory containing the training and validation datasets
#GEN_RESTORE_FILE="gen_weights.pth"  # Optional, path to the generator model weights to restore
#DISC_RESTORE_FILE="disc_weights.pth"  # Optional, path to the discriminator model weights to restore
GEN_RESTORE_FILE="g_best.pth.tar"  # Optional, path to the generator model weights to restore
DISC_RESTORE_FILE="d_best.pth.tar"  # Optional, path to the discriminator model weights to restore

# Training script execution
python train_and_evaluate.py \
    --test_dir $TEST_DIR \
    --data_dir $DATA_DIR \
    --gen_restore_file $GEN_RESTORE_FILE \
    --disc_restore_file $DISC_RESTORE_FILE