#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config-file-name>"
    exit 1
fi

CONFIG_PATH=$1
FULL_PATH="/home/"$(id -un)"/src/configs/${CONFIG_PATH}.yml"

echo "Training: $1"
echo "Running with config: $FULL_PATH"

export CUBLAS_WORKSPACE_CONFIG=:4096:8

python main.py --config-path $FULL_PATH