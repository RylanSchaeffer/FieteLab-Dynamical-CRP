#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 4                    # two cores
#SBATCH --mem=64G               # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-user=rylansch
#SBATCH --mail-type=FAIL

# write the executed command to the slurm output file for easy reproduction
# https://stackoverflow.com/questions/5750450/how-can-i-print-each-command-before-executing
set -x

source rncrp_venv/bin/activate

# don't remember what this does
export PYTHONPATH=.

# -u flushes output buffer immediately
python -u 05_swav_pretrained/run_one.py