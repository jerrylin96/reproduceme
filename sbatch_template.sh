#!/bin/bash
#SBATCH --job-name="boopboop"
#SBATCH --output="logs/srun-kerastuner-%j.%N.out"
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-16:1
#SBATCH --ntasks=1
#SBATCH --export=ALL
#SBATCH --account=atm200007p
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jerryl9@uci.edu
#SBATCH -t 5:00:00

cp -v /ocean/projects/atm200007p/jlin96/nnspreadtesting_2/overfitmepls/training/training_data/* /dev/shm
source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load anaconda3
conda activate wandb
python train.py
