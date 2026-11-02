#!/bin/bash
# Usage: sbatch scripts/slurm/train.sbatch configs/pi_multi_track_sequential.yaml
#SBATCH --job-name=thermano
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
CONFIG=${1:?config file required}
IMAGE=${THERMANO_IMAGE:-thermano.sif}

mkdir -p logs
echo "job ${SLURM_JOB_ID} on $(hostname) - config ${CONFIG} - commit $(git rev-parse --short HEAD)"
apptainer exec --nv "${IMAGE}" thermano train "${CONFIG}"
