#!/bin/bash
#SBATCH -p use-everything
#SBATCH -n 4
#SBATCH --mem=48G
#SBATCH --time=99:99:99
#SBATCH --mail-user=rylansch
#SBATCH --mail-type=FAIL

# write the executed command to the slurm output file for easy reproduction
# https://stackoverflow.com/questions/5750450/how-can-i-print-each-command-before-executing
set -x

source rncrp_venv/bin/activate

python3 rncrp/data/extract_swav_embeddings.py