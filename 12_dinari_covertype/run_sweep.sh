#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 1                    # two cores
#SBATCH --mem=1G                # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=FAIL

# source rncrp_venv/bin/activate
# wandb sweep 12_dinari_covertype/sweep_complete.yaml


for i in {1..5}
do
  sbatch 12_dinari_covertype/run_one.sh e8xxtb2x
  sleep 2
done
