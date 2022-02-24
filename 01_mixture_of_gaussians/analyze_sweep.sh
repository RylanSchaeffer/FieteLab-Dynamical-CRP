#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 1                    # one core
#SBATCH --mem=1G                # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-user=gkml
#SBATCH --mail-type=FAIL

export PYTHONPATH=.
python -u 01_mixture_of_gaussians/analyze_sweep.py       # -u flushes output buffer immediately

