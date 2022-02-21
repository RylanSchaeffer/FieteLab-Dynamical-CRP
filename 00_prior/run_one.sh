#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 2                    # two cores
#SBATCH --mem=24G               # RAM
#SBATCH --time=10:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-user=$USER
#SBATCH --mail-type=FAIL

run_one_results_dir_path=${1}
num_customer=${2}
num_mc_sample=${3}
alpha=${4}
beta=${5}
dynamics_str=${6}

# don't remember what this does
export PYTHONPATH=.

# -u flushes output buffer immediately
python -u 00_prior/run_one.py \
--results_dir_path="${run_one_results_dir_path}" \
--num_customer="${num_customer}" \
--num_mc_sample="${num_mc_sample}" \
--alpha="${alpha}" \
--beta="${beta}" \
--dynamics_str=${6}
