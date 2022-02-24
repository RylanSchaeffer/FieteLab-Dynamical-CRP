#!/bin/bash
#SBATCH -p fiete
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --gres=gpu:4
#SBATCH --mem=16G
#SBATCH --job-name=swav_4gpus
#SBATCH --time=99:99:99

#SBATCH --mail-type=FAIL

# write the executed command to the slurm output file for easy reproduction
# https://stackoverflow.com/questions/5750450/how-can-i-print-each-command-before-executing
set -x

source swav_venv/bin/activate

python3 -u rncrp/data/extract_swav_embeddings.py