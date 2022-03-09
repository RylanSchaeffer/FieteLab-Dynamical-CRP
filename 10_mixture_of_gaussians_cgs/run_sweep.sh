#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 1                    # two cores
#SBATCH --mem=1G                # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=FAIL


# Run this, then pipe sweep ID to each individual run
# source rncrp_venv/bin/activate
# wandb sweep 10_mixture_of_gaussians_cgs/sweep_debug.yaml

for i in {1..10}
do
  sbatch 10_mixture_of_gaussians_cgs/run_one.sh kavbcfgx
  sleep 5
done
