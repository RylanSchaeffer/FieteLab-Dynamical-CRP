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


def run_inference_alg(inference_alg_str: str,
                      observations: np.ndarray,
                      observations_times: np.ndarray,
                      concentration_param,
                      likelihood_model,
                      learning_rate):
    inference_alg_kwargs = dict()

    # RN-CRP
    if inference_alg_str == 'RN-CRP':
        inference_alg_fn = rn_crp

    # DP-GMM
    elif inference_alg_str.startswith('DP-GMM'):
        inference_alg_fn = dp_gmm

        substrs = inference_alg_str.split(' ')  # Parse parameters from algorithm string as needed
        num_initializations = int(substrs[2][1:])
        max_iters = int(substrs[4])

        inference_alg_kwargs['num_initializations'] = num_initializations
        inference_alg_kwargs['max_iter'] = max_iters

    # DP-Means
    elif inference_alg_str.startswith('DP-Means'):
        inference_alg_fn = dp_means

        if inference_alg_str.endswith('(offline)'):
            inference_alg_kwargs['num_passes'] = 8  # same as Kulis and Jordan

        elif inference_alg_str.endswith('(online)'):
            inference_alg_kwargs['num_passes'] = 1
        else:
            raise ValueError('Invalid DP Means')

    else:
        raise ValueError(f'Unknown inference algorithm: {inference_alg_str}')

    # Run inference algorithm
    inference_alg_results = inference_alg_fn(
        observations=observations,
        concentration_param=concentration_param,
        likelihood_model=likelihood_model,
        learning_rate=learning_rate,
        **inference_alg_kwargs)

    return inference_alg_results


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
