#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 1                    # two cores
#SBATCH --mem=1G                # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=FAIL


# Run this, then pipe sweep ID to each individual run
# source rncrp_venv/bin/activate
# wandb sweep 09_mixture_of_gaussians_debug/sweep_debug.yaml

for i in {1..20}
do
  sbatch 09_mixture_of_gaussians_debug/run_one.sh 1z40f04i
  sleep 2
done
