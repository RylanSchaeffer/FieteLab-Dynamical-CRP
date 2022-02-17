#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 2                    # two cores
#SBATCH --mem=16G               # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-user=gkml
#SBATCH --mail-type=FAIL

id=${1}

# update
export PYTHONPATH=.
export WANDB_CONFIG_DIR=/om2/user/gkml
export WANDB_API_KEY= ## TODO: FILL IN

source rncrp_venv/bin/activate

# write the executed command to the slurm output file for easy reproduction
# https://stackoverflow.com/questions/5750450/how-can-i-print-each-command-before-executing
set -x

wandb agent ## TODO: FILL IN