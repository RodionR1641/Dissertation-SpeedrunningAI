#!/bin/bash

#SBATCH --job-name=hello_world
#SBATCH --output=logs/slurm-%j.out
#SBATCH -c4 --mem=24g
#SBATCH --gres gpu:1
#SBATCH -p cs -q csug

source /usr2/share/gpu.sbatch

#torchrun main.py --data-path /db/psyrr4/breakout/

LOG_DIR="/cs/home/psyrr4/Code/Code/logs"
mkdir -p $LOG_DIR

LOG_FILE="${LOG_DIR}/train_${SLURM_JOB_ID}.log"

PORT_RANGE_START=29500
PORT_RANGE_END=29899

# Find an available port
for port in $(seq $PORT_RANGE_START $PORT_RANGE_END); do
    if ! ss -tuln | grep -q ":$port "; then
        MASTER_PORT=$port
        break
    fi
done

if [ -z "$MASTER_PORT" ]; then
    echo "No available ports in range $PORT_RANGE_START-$PORT_RANGE_END"
    exit 1
fi

echo "port is: $MASTER_PORT"

module load nvidia/cuda-11.2
module load nvidia/cudnn-v8.1.1.33-forcuda11.0-to-11.2

echo "Allocated GPUs: $SLURM_GPUS_ON_NODE"

#torchrun --master_port=$MASTER_PORT helloworld.py --data-path /db/psyrr4/breakout/ > $LOG_FILE 2>&1
python3 helloworld.py --master_port=$MASTER_PORT --data-path /db/psyrr4/breakout/ > $LOG_FILE 2>&1
