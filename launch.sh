#!/bin/bash
#SBATCH -p all # partition (queue) 
#SBATCH -c 4 # number of cores 
#SBATCH --mem=100G 
#SBATCH --propagate=NONE 
# IMPORTANT for long jobs 
#SBATCH -t 1-00:00 # time (D-HH:MM) 
#SBATCH --mail-user=arie.itovitch@mail.mcgill.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=output.log 
#SBATCH --error=error.log 
#SBATCH --qos=comp579-0gpu-4cpu-72h 
#SBATCH --account=winter2025-comp579


module load cuda/cuda-12.6
module load python/3.10

pip3 cache purge
pip3 install -r requirements.txt

python3 a3.py
