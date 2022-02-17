#!/bin/bash
#SBATCH -p use-everything
#SBATCH -n 2                    # two cores
#SBATCH --mem=32G               # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-user=rylansch
#SBATCH --mail-type=FAIL

source rncrp_venv/bin/activate

# don't remember what this does
export PYTHONPATH=.

# -u flushes output buffer immediately
python -u 05_swav_pretrained/run_one.py