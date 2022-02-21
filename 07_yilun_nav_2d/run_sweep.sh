#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 1                    # two cores
#SBATCH --mem=1G                # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-user=rylansch
#SBATCH --mail-type=FAIL


# Run this, then pipe sweep ID to each individual run
# source rncrp_venv/bin/activate
# wandb sweep 07_yilun_nav_2d/sweep_complete.yaml

for i in {1..10}
do
  sbatch 07_yilun_nav_2d/run_one.sh nhj1s21j
  sleep 5
done
