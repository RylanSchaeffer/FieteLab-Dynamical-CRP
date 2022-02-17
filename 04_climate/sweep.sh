#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 1                    # two cores
#SBATCH --mem=4G                # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-user=gkml
#SBATCH --mail-type=FAIL


# Run this, then pipe ID to each individual run
# export WANDB_CONFIG_DIR=/om2/user/gkml
# export WANDB_API_KEY= ## TODO: FILL IN
# source rncrp_venv/bin/activate
# wandb sweep exp2_climate/sweep.yaml

for i in {1..5}
do
  sbatch exp2_climate/run_one.sh 0flwonho ## <-- TODO: REPLACE NAME??
  sleep 5
done