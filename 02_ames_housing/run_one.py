"""
Perform inference in Ames Housing for the specified inference algorithm
and model parameter.

Example usage:

02_ames_housing/run_one.py
"""

import argparse
import joblib
import logging
import numpy as np
import os
import torch
import wandb

import rncrp.data.real
import rncrp.helpers.dynamics
import rncrp.helpers.run
import rncrp.metrics

config_defaults = {
    'inference_alg_str': 'VI-GMM',
    # 'inference_alg_str': 'DP-Means (Offline)',
    # 'inference_alg_str': 'RN-CRP',
    'dynamics_str': 'hyperbolic',
    'dynamics_a': 1.,
    'dynamics_b': 1.,
    'dynamics_c': 1.,
    'dynamics_omega': np.pi / 2.,
    'alpha': 0.1,
    'beta': 0.,
    'centroids_prior_cov_prefactor': 5.,
    'likelihood_cov_prefactor': 50.,
    'repeat_idx': 0,
}

wandb.init(project='dcrp-ames-housing',
           config=config_defaults)
config = wandb.config

print(f'Running:')
for key, value in config.items():
    print(key, ' : ', value)

# Set paths and create if necessary.
exp_dir = '02_ames_housing'
results_dir_path = os.path.join(exp_dir, 'results')
os.makedirs(results_dir_path, exist_ok=True)
inf_alg_results_path = os.path.join(results_dir_path,
                                    f'id={wandb.run.id}.joblib')
wandb.log({'inf_alg_results_path': inf_alg_results_path},
          step=0)

# Set seeds.
rncrp.helpers.run.set_seed(config['repeat_idx'])

# Load data and permute order of data based on seed.
ames_housing_data = rncrp.data.real.load_dataset_ames_housing_2011()
n_obs = ames_housing_data['observations'].shape[0]
permutation = np.random.permutation(np.arange(n_obs))
ames_housing_data['observations'] = \
    ames_housing_data['observations'][permutation]
ames_housing_data['labels'] = \
    ames_housing_data['labels'][permutation]
print('Randomly permuted the order of 2016 Cancer Gene Expression data.')

# Compute the correct parameters, depending on the dynamics
if config['dynamics_str'] == 'step':
    dynamics_params = {}
elif config['dynamics_str'] == 'exp':
    dynamics_params = {}
elif config['dynamics_str'] == 'hyperbolic':
    dynamics_params = {}
elif config['dynamics_str'] == 'sinusoid':
    dynamics_params = {}
else:
    raise NotImplementedError


gen_model_params = {
    'mixing_params': {
        'alpha': config['alpha'],
        'beta': config['beta'],
        'dynamics_str': config['dynamics_str'],
        'dynamics_params': dynamics_params
    },
    'feature_prior_params': {
        'centroids_prior_cov_prefactor': config['centroids_prior_cov_prefactor']
    },
    'likelihood_params': {
        'distribution': 'multivariate_normal',
        'likelihood_cov_prefactor': config['likelihood_cov_prefactor']
    }
}

observations_times = ames_housing_data['observations']
observations = ames_housing_data['observations'].copy()

inference_alg_results = rncrp.helpers.run.run_inference_alg(
    inference_alg_str=config['inference_alg_str'],
    observations=observations,
    observations_times=observations_times,
    gen_model_params=gen_model_params,
)


sum_sqrd_distances = rncrp.metrics.compute_sum_of_squared_distances_to_nearest_center(
    X=mixture_model_results['observations'],
    centroids=inference_alg_results['inference_alg'].centroids_after_last_obs())
inference_alg_results['training_reconstruction_error'] = sum_sqrd_distances
wandb.log({'training_reconstruction_error': sum_sqrd_distances}, step=0)


data_to_store = dict(
    config=dict(config),  # Need to convert WandB config to proper dict
    inference_alg_results=inference_alg_results,
    )

joblib.dump(data_to_store,
            filename=inf_alg_results_path)

print('Finished run.')