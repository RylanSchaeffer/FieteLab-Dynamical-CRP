#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 1                    # two cores
#SBATCH --mem=1G                # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=FAIL


# Run this, then pipe sweep ID to each individual run
# source rncrp_venv/bin/activate
# wandb sweep 03_mixture_of_vonmises_fisher/sweep_complete.yaml

for i in {1..2}
do
  sbatch 03_mixture_of_vonmises_fisher/run_one.sh 2yd5qdao
  sleep 5
done
