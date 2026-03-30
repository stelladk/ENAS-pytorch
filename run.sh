#!/bin/bash

#SBATCH --job-name=ENAS
#SBATCH --output=slurm/slurm-%x-%A_%a.out
#SBATCH --time=5-00:10:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -p tau
#SBATCH --exclude=margpu018,margpu021
#SBATCH --array=[1-5]

# SBATCH -w margpu024


STAGGER_SECONDS=5
SLEEP_TIME=$(( (SLURM_ARRAY_TASK_ID - 1) * STAGGER_SECONDS ))
echo "Array task ${SLURM_ARRAY_TASK_ID}: sleeping ${SLEEP_TIME}s before starting"
sleep "${SLEEP_TIME}"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

if [ "${SLURM_ARRAY_JOB_ID}" ] ; then
    JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
    JOB_ID="${SLURM_JOB_ID}"
fi
echo -e "JOB ID = ${JOB_ID}"
echo -e "NODE NAME = ${SLURMD_NODENAME}"

echo -e "DATASET = ${NAS_DATASET}"
echo -e "CLASSES = ${NAS_CLASSES}"

python train_search.py --dataset "${NAS_DATASET}" --num_classes "${NAS_CLASSES}"

