#!/bin/bash

#SBATCH -p all # partition (queue) 
#SBATCH -c 4 # number of cores 
#SBATCH --mem=4G 
#SBATCH --propagate=NONE 
#SBATCH -t 3-00:00 # time (D-HH:MM) 
#SBATCH --output=output.log 
#SBATCH --error=error.log 
#SBATCH --qos=comp579-0gpu-4cpu-72h 
#SBATCH --account=winter2025-comp579
#SBATCH --mail-user=arie.itovitch@mail.mcgill.ca
#SBATCH --mail-type=BEGIN,END,FAIL

module load cuda/cuda-12.6 
#module load python/3.10
#pip cache purge
#pip3 install --target=/home/2022/aitovi/my_python_env/lib/python3.10/site-packages -r requirements.txt

#pip cache purge
#pip install -r requirements.txt

python3 a3.py
