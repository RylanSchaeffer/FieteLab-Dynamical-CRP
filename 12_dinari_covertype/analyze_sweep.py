import numpy as np
import os
import pandas as pd

import rncrp.data.real_tabular
from rncrp.helpers.analyze import download_wandb_project_runs_configs
import plot_dinari_covertype


exp_dir = '12_dinari_covertype'
results_dir = os.path.join(exp_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
wandb_sweep_path = "rylan/rncrp-dinari-covertype"
sweep_names = [
    'e8xxtb2x',
]
sweep_names_str = ','.join(sweep_names)
print(f'Analyzing sweeps {sweep_names_str}')
sweep_results_dir_path = os.path.join(results_dir, sweep_names_str)
os.makedirs(sweep_results_dir_path, exist_ok=True)

all_inf_algs_results_df = download_wandb_project_runs_configs(
    wandb_project_path=wandb_sweep_path,
    data_dir=results_dir,
    sweep_ids=sweep_names,
    finished_only=True,
    refresh=False)

print(f"Number of runs: {all_inf_algs_results_df.shape[0]} for sweep={sweep_names_str}")

plot_dinari_covertype.plot_analyze_all_inf_algs_results(
    all_inf_algs_results_df=all_inf_algs_results_df,
    plot_dir=sweep_results_dir_path,
)

# Load Dinari cover type data.
dinari_covertype_data = rncrp.data.real_tabular.load_dataset_dinari_covertype_2022()

n_samples = all_inf_algs_results_df['n_samples'].unique()[0]
plot_dinari_covertype.plot_dinari_covertype_labels(
    labels=dinari_covertype_data['labels'][:n_samples],
    plot_dir=sweep_results_dir_path,
)

print(f'Finished 12_dinari_covertype/analyze_sweep.py for sweep={sweep_names}.')
