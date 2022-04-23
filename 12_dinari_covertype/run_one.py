"""
Perform inference in Dinari's preprocessed CoverType dataset (https://archive.ics.uci.edu/ml/datasets/covertype)
for the specified inference algorithm and model (hyper)parameters.

Example usage:

12_dinari_covertype/run_one.py
"""

import argparse
import joblib
import logging
import numpy as np
import os
import torch
import wandb

import rncrp.data.real_tabular
import rncrp.helpers.dynamics
import rncrp.helpers.run
import rncrp.metrics

config_defaults = {
    # 'inference_alg_str': 'VI-GMM',
    # 'inference_alg_str': 'DP-Means (Offline)',
    'inference_alg_str': 'Dynamical-CRP',
    'dynamics_str': 'exp',
    'dynamics_a': 1.,
    'dynamics_b': 1.,
    'dynamics_c': 1.,
    'dynamics_omega': np.pi / 2.,
    'alpha': 0.5,
    'beta': 0.,
    'centroids_prior_cov_prefactor': 5.,
    'likelihood_cov_prefactor': 50.,
    'repeat_idx': 0,
}

wandb.init(project='dcrp-dinari-covertype',
           config=config_defaults)
config = wandb.config

print(f'Running:')
for key, value in config.items():
    print(key, ' : ', value)

# Set paths and create if necessary.
exp_dir = '12_dinari_covertype'
results_dir_path = os.path.join(exp_dir, 'results')
os.makedirs(results_dir_path, exist_ok=True)
inf_alg_results_path = os.path.join(results_dir_path,
                                    f'id={wandb.run.id}.joblib')
wandb.log({'inf_alg_results_path': inf_alg_results_path},
          step=0)

# Set seeds.
rncrp.helpers.run.set_seed(seed=config['repeat_idx'])

# Load data and sort based on date.
dinari_covertype_data = rncrp.data.real_tabular.load_dataset_dinari_covertype_2022()
observations = dinari_covertype_data['observations']
observations_times = 1 + np.arange(len(observations))
true_cluster_assignments = dinari_covertype_data['labels']

# Compute and log the number of true clusters.
wandb.log({'n_clusters': len(np.unique(true_cluster_assignments))}, step=0)

# Compute the correct parameters, depending on the dynamics
if config['dynamics_str'] == 'sinusoid':
    dynamics_params = {'omega': config['dynamics_omega']}
elif config['dynamics_str'] == 'step':
    dynamics_params = {'a': config['dynamics_a'], 'b': 0}
elif config['dynamics_str'] == 'exp':
    dynamics_params = {'a': config['dynamics_a'], 'b': config['dynamics_b']}
elif config['dynamics_str'] == 'hyperbolic':
    dynamics_params = {'c': config['dynamics_c']}
else:
    raise NotImplementedError


gen_model_params = {
    'mixing_params': {
        'alpha': config['alpha'],
        'beta': config['beta'],
        'dynamics_str': config['dynamics_str'],
        'dynamics_params': dynamics_params
    },
    'component_prior_params': {
        'centroids_prior_cov_prefactor': config['centroids_prior_cov_prefactor']
    },
    'likelihood_params': {
        'distribution': 'multivariate_normal',
        'likelihood_cov_prefactor': config['likelihood_cov_prefactor']
    }
}

inference_alg_results = rncrp.helpers.run.run_inference_alg(
    inference_alg_str=config['inference_alg_str'],
    observations=observations,
    observations_times=observations_times,
    gen_model_params=gen_model_params,
)

scores, map_cluster_assignments = rncrp.metrics.compute_predicted_clusters_scores(
    cluster_assignment_posteriors=inference_alg_results['cluster_assignment_posteriors'],
    true_cluster_assignments=true_cluster_assignments,
)
wandb.log(scores, step=0)
inference_alg_results.update(scores)
inference_alg_results['map_cluster_assignments'] = map_cluster_assignments

data_to_store = dict(
    config=dict(config),  # Need to convert WandB config to proper dict
    inference_alg_results=inference_alg_results,
    scores=scores)

joblib.dump(data_to_store,
            filename=inf_alg_results_path)

print('Finished run.')
