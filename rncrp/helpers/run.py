import logging
import numpy as np
import os
import pandas as pd
import sys
import torch
import wandb


def create_logger(run_dir):

    logging.basicConfig(
        filename=os.path.join(run_dir, 'logging.log'),
        level=logging.DEBUG)

    logging.info('Logger created successfully')

    # also log to std out
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(console_handler)

    # disable matplotlib font warnings
    logging.getLogger("matplotlib").setLevel(logging.ERROR)


def download_wandb_project_runs_results(wandb_project_path: str,
                                        sweep_name: str = None,
                                        ) -> pd.DataFrame:

    # Download sweep results
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs(path=wandb_project_path)

    sweep_results_list = []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary = run.summary._json_dict

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        summary.update(
            {k: v for k, v in run.config.items()
             if not k.startswith('_')})

        summary.update({'State': run.state,
                        'Sweep': run.sweep.id if run.sweep is not None else None})
        # .name is the human-readable name of the run.
        summary.update({'run_name': run.name})
        sweep_results_list.append(summary)

    sweep_results_df = pd.DataFrame(sweep_results_list)

    # Keep only finished runs
    sweep_results_df = sweep_results_df[sweep_results_df['State'] == 'finished']

    if sweep_name is not None:
        sweep_results_df = sweep_results_df[sweep_results_df['Sweep'] == sweep_name]

    sweep_results_df = sweep_results_df.copy()

    return sweep_results_df


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
