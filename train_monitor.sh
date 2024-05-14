#!/bin/bash

# Script to run distributed training
TRAINING_SCRIPT="./train_distributed.sh"

# GPU monitoring script
MONITOR_SCRIPT="./monitor_gpu.sh"

# Log file for GPU usage
GPU_LOG="gpu_usage.log"

# Start the distributed training script
nohup bash $TRAINING_SCRIPT > training_output.log 2>&1 &

# Start the GPU monitoring script
nohup bash $MONITOR_SCRIPT $GPU_LOG > monitor_output.log 2>&1 &

echo "Training and GPU monitoring have been started."
