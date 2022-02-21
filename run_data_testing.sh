#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 4                    # two cores
#SBATCH --mem=32G               # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-user=$USER
#SBATCH --mail-type=FAIL

source rncrp_venv/bin/activate

# don't remember what this does
export PYTHONPATH=.

# -u flushes output buffer immediately
python -u rncrp/data/real.py