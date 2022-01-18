#!/bin/bash
#SBATCH -p use-everything
#SBATCH -n 1                    # two cores
#SBATCH --mem=4G                # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-user=rylansch
#SBATCH --mail-type=FAIL


# Run this, then pipe sweep ID to each individual run
# export WANDB_CONFIG_DIR=/om2/user/rylansch
# export WANDB_API_KEY=51a0a43a1b4ba9981701d60c5f6887cd5bf9e03e
# source rncrp_venv/bin/activate
# wandb sweep 01_mixture_of_gaussians/sweep_quick.yaml
# wandb sweep 01_mixture_of_gaussians/sweep_complete.yaml

for i in {1..5}
do
  sbatch 02_heald_sensorimotor_exp1a/run_one.sh fsaifrzx
  sleep 5
done
