#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 1                    # two cores
#SBATCH --mem=1G                # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=FAIL


# Run this, then pipe sweep ID to each individual run
# source rncrp_venv/bin/activate
# wandb sweep 01_mixture_of_gaussians/sweep_quick.yaml
# wandb sweep 01_mixture_of_gaussians/sweep_complete.yaml

for i in {1..10}
do
  sbatch 01_mixture_of_gaussians/run_one.sh psmahisk
  sleep 5
done
