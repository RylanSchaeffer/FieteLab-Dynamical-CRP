#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 1                    # one core
#SBATCH --mem=32G               # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=FAIL

export PYTHONPATH=.
python -u 05_swav_pretrained/analyze_sweep.py       # -u flushes output buffer immediately

