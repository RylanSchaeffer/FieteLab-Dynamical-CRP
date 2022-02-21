#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 2                    # two cores
#SBATCH --mem=10G               # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-user=$USER
#SBATCH --mail-type=FAIL

id=${1}

# update
export PYTHONPATH=.
export WANDB_CONFIG_DIR=/om2/user/rylansch
export WANDB_API_KEY=51a0a43a1b4ba9981701d60c5f6887cd5bf9e03e

source rncrp_venv/bin/activate

# write the executed command to the slurm output file for easy reproduction
# https://stackoverflow.com/questions/5750450/how-can-i-print-each-command-before-executing
set -x

wandb agent rylan/dcrp-yilun-nav2d/${id}
