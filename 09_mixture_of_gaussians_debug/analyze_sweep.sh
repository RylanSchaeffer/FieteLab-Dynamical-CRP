#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 1                    # one core
#SBATCH --mem=16G               # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=FAIL

export PYTHONPATH=.
python -u 09_mixture_of_gaussians_debug/analyze_sweep.py       # -u flushes output buffer immediately

