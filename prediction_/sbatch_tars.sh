#! /bin/bash

# Sbatch options
#SBATCH -J bambino
##SBATCH -p dbc_pmo
##SBATCH -p common
#SBATCH -p dbc
#SBATCH --qos=fast
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10000MB



# Constants
logs_folder="./logs/"
# args_file="arguments.dat"

# # Read command line arguments from file
# argument=`awk "NR==${SLURM_ARRAY_TASK_ID}" $args_file`

# Launch srun with these argument sequence
module load Python/3.10.7
echo $argument
srun -o "${logs_folder}log_job_${SLURM_ARRAY_TASK_ID}.out" -e "${logs_folder}log_job_${SLURM_ARRAY_TASK_ID}.err" -J "${SLURM_ARRAY_TASK_ID}" python3 job_manager.py
