#!/bin/bash

# 定义参数
PARAMS_PATH="tests/truecondW1/params.json"
RESTORE_FILE="g_best.pth.tar"
LOCATION_CSV="location.CSV"
CONDITION="160-120-2200"
OUTPUT_FOLDER="output"
COORDINATE_TYPE="polar"
DATA_DIR="data_raw/data/split_data"
TEST_DIR="tests/truecondW1"

# 运行推理脚本
python inference.py \
    --params_path $PARAMS_PATH \
    --restore_file $RESTORE_FILE \
    --location_csv $LOCATION_CSV \
    --condition $CONDITION \
    --output_folder $OUTPUT_FOLDER \
    --coordinate_type $COORDINATE_TYPE \
    --data_dir $DATA_DIR \
    --test_dir $TEST_DIR
