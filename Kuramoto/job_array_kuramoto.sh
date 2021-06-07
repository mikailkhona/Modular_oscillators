#!/bin/bash
#SBATCH --job-name=kuramototest
#SBATCH --mem=25gb
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output=/om/user/facosta/Kuramoto/simulation/outputlogs/kuramototest_%A_%a.log
#SBATCH --partition=fiete
#SBATCH --priority=fiete
#SBATCH --ntasks=1
#SBATCH --nodes=4
#SBATCH --gres=gpu:GEFORCERTX2080:1
#SBATCH --array=100-200:5 


cd /om/user/facosta/Kuramoto/

module load openmind/anaconda/3-2019.10
python3 -u Kuramoto.py 2500 $SLURM_ARRAY_TASK_ID 0 10 False False 0 0 60 0.001 "linear" 500 0.01


