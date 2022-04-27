#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 1                    # two cores
#SBATCH --mem=1G                # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=FAIL


# Run this, then pipe sweep ID to each individual run
# source rncrp_venv/bin/activate
# wandb sweep 05_swav_pretrained/sweep_complete.yaml
# wandb sweep 05_swav_pretrained/sweep_quick.yaml

for i in {1..12}
do
  sbatch 05_swav_pretrained/run_one.sh 6yflkbeq
  sleep 2
done
