#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 2                    # one core
#SBATCH --mem=16G               # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=FAIL

export PYTHONPATH=.
python -u 01_mixture_of_gaussians/analyze_sweep.py       # -u flushes output buffer immediately

