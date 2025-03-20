
#!/bin/bash
#SBATCH -w gpu-grad-01
#SBATCH --ntasks 8
#SBATCH --mem 32GB
#SBATCH -t 0-2:00
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err
#SBATCH --mail-user=<your_name>@mail.mcgill.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=output.log 
#SBATCH --error=error.log 
#SBATCH --qos=comp579-0gpu-4cpu-72h 
#SBATCH --account=winter2025-comp579

module load cuda/cuda-12.6
pip3 cache purge
pip3 install -r requirements.txt
python3 -c "import sys; print(sys.path)"

python3 a3.py