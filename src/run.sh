#!/bin/bash

NUM_GPUS=4
CONFIG="config/config_test.yaml"

export PYTHONPATH=$(pwd)

python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS src/train_distributed.py --config $CONFIG