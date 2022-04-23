#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 2                    # one core
#SBATCH --mem=16G               # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=FAIL

source rncrp_venv/bin/activate

export PYTHONPATH=.

python -u 07_yilun_nav_2d/analyze_sweep.py       # -u flushes output buffer immediately

