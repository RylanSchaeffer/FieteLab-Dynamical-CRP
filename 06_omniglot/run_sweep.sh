#!/bin/bash
#SBATCH -p normal
#SBATCH -n 1                    # two cores
#SBATCH --mem=1G                # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=FAIL


# Run this, then pipe sweep ID to each individual run
# source rncrp_venv/bin/activate
# wandb sweep 06_omniglot/sweep_complete.yaml

for i in {1..12}
do
  sbatch 06_omniglot/run_one.sh opi7cxik
  sleep 5
done
