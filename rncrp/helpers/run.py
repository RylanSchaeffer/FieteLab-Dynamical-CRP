import logging
import numpy as np
import os
import pandas as pd
import sys
import torch
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from typing import Dict, List, Tuple
import wandb
import tensorflow as tf
from typing import Union

from rncrp.inference import CollapsedGibbsSampler, CollapsedGibbsSamplerNew, DPMeans,\
    DynamicalCRP, KMeans, RecursiveCRP, VariationalInferenceGMM


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


def run_inference_alg(inference_alg_str: str,
                      observations: Union[np.ndarray, DataLoader],
                      observations_times: np.ndarray,
                      gen_model_params: Dict[str, Dict[str, float]],
                      inference_alg_kwargs: Dict = None):

    if inference_alg_kwargs is None:
        inference_alg_kwargs = dict()

    if inference_alg_str == 'CollapsedGibbsSampler':

        # inference_alg = CollapsedGibbsSampler(
        #     gen_model_params=gen_model_params,
        #     **inference_alg_kwargs
        # )
        inference_alg = CollapsedGibbsSamplerNew(
            gen_model_params=gen_model_params,
            **inference_alg_kwargs
        )

    elif inference_alg_str.startswith('Dynamical-CRP'):

        # TODO: Refactor model names to not contain parameters, you nit wit.
        if inference_alg_str.endswith('(Cutoff=1e-2)'):
            inference_alg_kwargs['cutoff'] = 1e-2
        elif inference_alg_str.endswith('(Cutoff=1e-3)'):
            inference_alg_kwargs['cutoff'] = 1e-3
        elif inference_alg_str.endswith('(Cutoff=1e-4)'):
            inference_alg_kwargs['cutoff'] = 1e-4

        inference_alg = DynamicalCRP(
            gen_model_params=gen_model_params,
            **inference_alg_kwargs)

    elif inference_alg_str.startswith('DP-Means'):
        if inference_alg_kwargs is None:
            inference_alg_kwargs = dict()

        if inference_alg_str.endswith('(Offline)'):
            inference_alg_kwargs['max_iter'] = 8  # Matching Kulis and Jordan.
        elif inference_alg_str.endswith('(Online)'):
            inference_alg_kwargs['max_iter'] = 1
        else:
            raise ValueError('Invalid DP Means')

        if 'lambda' not in gen_model_params['mixing_params']:
            # 20 is arbitrary. Just want a reasonable range.
            # gen_model_params['mixing_params']['lambda'] = 20. / gen_model_params['mixing_params']['alpha']
            # gen_model_params['mixing_params']['lambda'] = 20. / np.sqrt(gen_model_params['mixing_params']['alpha'])
            assert gen_model_params['mixing_params']['alpha'] > 0.
            # 1/log(alpha) seems to work best, of all scalings.
            # 50 is better prefactor than 20.
            gen_model_params['mixing_params']['lambda'] = 20. / np.log(gen_model_params['mixing_params']['alpha'])

        inference_alg = DPMeans(
            gen_model_params=gen_model_params,
            **inference_alg_kwargs)

    elif inference_alg_str.startswith('K-Means'):

        if inference_alg_kwargs is None:
            inference_alg_kwargs = dict()

        if inference_alg_str.endswith('(Offline)'):
            inference_alg_kwargs['max_iter'] = 8  # Matching DP Means
        elif inference_alg_str.endswith('(Online)'):
            inference_alg_kwargs['max_iter'] = 1
            inference_alg_kwargs['num_initializations'] = 1
        else:
            raise ValueError('Invalid KMeans')

        inference_alg = KMeans(
            gen_model_params=gen_model_params,
            **inference_alg_kwargs)

    elif inference_alg_str == 'Recursive-CRP':
        if inference_alg_kwargs is None:
            inference_alg_kwargs = dict()

        inference_alg = RecursiveCRP(
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


def select_indices_given_desired_cluster_assignments_and_labels(
        desired_cluster_assignments: np.ndarray,
        labels: np.ndarray) -> np.ndarray:

    """
    If we're given the sequence of desired cluster assignments e.g. 0, 1, 0, 0, 1, 2...
    And the true label per class e.g. 0,...,0,1,...,1,2,....,2 etc.

    desired_cluster_assignments: Shape (num desired data,)
    labels: Shape: (total num observations,)
    """

    # First, find out how many unique classes are requested and how many data from each.
    cluster_ids, n_data_per_cluster_id = np.unique(
        desired_cluster_assignments,
        return_counts=True)
    num_clusters = len(cluster_ids)

    labels_df = pd.DataFrame({'labels': labels})

    indices = np.full_like(desired_cluster_assignments, fill_value=-1)
    for desired_cluster_assignment in desired_cluster_assignments:
        print(10)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)
