#!/bin/bash

NUM_GPUS=4
CONFIG="config/config_250721.yaml"

python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS train_distributed.py --config $CONFIG