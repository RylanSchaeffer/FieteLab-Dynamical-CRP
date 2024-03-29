#!/bin/bash
#SBATCH -p normal
#SBATCH -n 1                    # two cores
#SBATCH --mem=1G                # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=FAIL


# Run this, then pipe sweep ID to each individual run
# source rncrp_venv/bin/activate
# wandb sweep 07_yilun_nav_2d/sweep_complete.yaml

for i in {1..12}
do
  sbatch 07_yilun_nav_2d/run_one.sh 4reny29o
  sleep 2
done
