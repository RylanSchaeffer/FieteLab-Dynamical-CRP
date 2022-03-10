#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 1                    # two cores
#SBATCH --mem=1G                # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=FAIL


# Run this, then pipe sweep ID to each individual run
# source rncrp_venv/bin/activate
# wandb sweep 03_mixture_of_vonmises_fisher/sweep_complete.yaml
# wandb sweep 03_mixture_of_vonmises_fisher/sweep_quick.yaml

for i in {1..12}
do
  sbatch 03_mixture_of_vonmises_fisher/run_one.sh axi0h0qe
  sleep 5
done
