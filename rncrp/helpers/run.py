import logging
import numpy as np
import os
import pandas as pd
from sklearn.cluster import KMeans
import sys
import torch
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from typing import Dict, List, Tuple
import wandb
import tensorflow as tf
from typing import Union

from rncrp.inference import DPMeans, DynamicalCRP, VariationalInferenceGMM


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
                                        sweep_id: str = None,
                                        ) -> pd.DataFrame:

    # Download sweep results
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    if sweep_id is None:
        runs = api.runs(path=wandb_project_path)
    else:
        runs = api.runs(path=wandb_project_path,
                        filters={"Sweep": sweep_id})

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
    finished_runs = sweep_results_df['State'] == 'finished'
    print(f"% of successfully finished runs: {finished_runs.mean()}")
    sweep_results_df = sweep_results_df[finished_runs]

    if sweep_id is not None:
        sweep_results_df = sweep_results_df[sweep_results_df['Sweep'] == sweep_id]

    # Ensure we aren't working with a slice.
    sweep_results_df = sweep_results_df.copy()

    return sweep_results_df


def run_inference_alg(inference_alg_str: str,
                      observations: Union[np.ndarray, DataLoader],
                      observations_times: np.ndarray,
                      gen_model_params: Dict[str, Dict[str, float]],
                      inference_alg_kwargs: Dict = None):

    if inference_alg_str == 'Dynamical-CRP':
        if inference_alg_kwargs is None:
            inference_alg_kwargs = dict()
        inference_alg = DynamicalCRP(
            gen_model_params=gen_model_params,
            **inference_alg_kwargs)

    elif inference_alg_str.startswith('DP-Means'):
        if inference_alg_kwargs is None:
            inference_alg_kwargs = dict()
        if inference_alg_str.endswith('(Offline)'):
            inference_alg_kwargs['max_iter'] = 8  # Matching by Kulis and Jordan.
        elif inference_alg_str.endswith('(Online)'):
            inference_alg_kwargs['max_iter'] = 1
        else:
            raise ValueError('Invalid DP Means')

        if 'lambda' not in gen_model_params['mixing_params']:
            # 20 is arbitrary. Just want a reasonable range.
            gen_model_params['mixing_params']['lambda'] = 20. / gen_model_params['mixing_params']['alpha']

        inference_alg = DPMeans(
            gen_model_params=gen_model_params,
            **inference_alg_kwargs)

    elif inference_alg_str == 'VI-GMM':
        if inference_alg_kwargs is None:
            inference_alg_kwargs = dict()

        inference_alg = VariationalInferenceGMM(
            gen_model_params=gen_model_params,
            **inference_alg_kwargs)

    else:
        raise ValueError(f'Unknown inference algorithm: {inference_alg_str}')

    # Run inference algorithm
    # time using timer because https://stackoverflow.com/a/25823885/4570472
    start_time = timer()
    inference_alg_results = inference_alg.fit(
        observations=observations,
        observations_times=observations_times,
    )
    stop_time = timer()
    runtime = stop_time - start_time
    inference_alg_results['Runtime'] = runtime
    wandb.log({'Runtime': runtime,
               'Num Inferred Clusters': inference_alg_results['num_inferred_clusters']}, step=0)

    # Add inference alg object to results, for later generating predictions
    inference_alg_results['inference_alg'] = inference_alg

    return inference_alg_results


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)
