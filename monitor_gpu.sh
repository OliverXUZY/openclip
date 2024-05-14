#!/bin/bash

# File to log the output
LOG_FILE="gpu_usage.log"

# Interval in seconds (e.g., 3600 seconds = 1 hour)
INTERVAL=360

while true
do
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')" >> $LOG_FILE
    nvidia-smi >> $LOG_FILE
    sleep $INTERVAL
done
