"""
Perform inference in a synthetic mixture of Gaussians for the specified inference
algorithm and dynamics string.

Example usage:

01_mixture_of_gaussians/run_one.py
"""

import argparse
import joblib
import logging
import numpy as np
import os
import torch
import wandb

# import plot
import rncrp.data.synthetic
import rncrp.helpers.dynamics
import rncrp.helpers.run
import rncrp.metrics

config_defaults = {
    # 'inference_alg_str': 'VI-GMM',
    # 'inference_alg_str': 'DP-Means (Offline)',
    'inference_alg_str': 'RN-CRP',
    'dynamics_str': 'hyperbolic',
    'dynamics_a': 1.,
    'dynamics_b': 1.,
    'dynamics_c': 1.,
    'dynamics_omega': np.pi / 2.,
    'n_samples': 130,
    'n_features': 10,
    'n_clusters': 40,
    'alpha': 0.1,
    'beta': 0.,
    'centroids_prior_cov_prefactor': 5.,
    'likelihood_cov_prefactor': 50.,
    'repeat_idx': 0,
}

wandb.init(project='dcrp-mixture-of-gaussians',
           config=config_defaults)
config = wandb.config

print(f'Running:')
for key, value in config.items():
    print(key, ' : ', value)

# determine paths
exp_dir = '01_mixture_of_gaussians'
results_dir_path = os.path.join(exp_dir, 'results')
os.makedirs(results_dir_path, exist_ok=True)
inference_alg_results_path = os.path.join(results_dir_path,
                                          f'id={wandb.run.id}.joblib')

# set seeds
rncrp.helpers.run.set_seed(config['repeat_idx'])

mixture_model_results = rncrp.data.synthetic.sample_mixture_model(
    num_obs=config['n_samples'],
    obs_dim=config['n_features'],
    mixing_prior_str='rncrp',
    mixing_distribution_params={'alpha': config['alpha'],
                                'beta': config['beta'],
                                'dynamics_str': config['dynamics_str']},
    component_prior_str='gaussian',
    component_prior_params={'centroids_prior_cov_prefactor': config['centroids_prior_cov_prefactor'],
                            'likelihood_cov_prefactor': config['likelihood_cov_prefactor']})

gen_model_params = {
    'mixing_params': {
        'alpha': config['alpha'],
        'beta': config['beta'],
        'dynamics_str': config['dynamics_str'],
        'dynamics_params': mixture_model_results['dynamics_params']
    },
    'feature_prior_params': {
        'centroids_prior_cov_prefactor': config['centroids_prior_cov_prefactor']
    },
    'likelihood_params': {
        'distribution': 'multivariate_normal',
        'likelihood_cov_prefactor': config['likelihood_cov_prefactor']
    }
}

inference_alg_results = rncrp.helpers.run.run_inference_alg(
    inference_alg_str=config['inference_alg_str'],
    observations=mixture_model_results['observations'],
    observations_times=mixture_model_results['observations_times'],
    gen_model_params=gen_model_params,
)

scores, map_cluster_assignments = rncrp.metrics.compute_predicted_clusters_scores(
    cluster_assignment_posteriors=inference_alg_results['cluster_assignment_posteriors'],
    true_cluster_assignments=mixture_model_results['cluster_assignments'],
)

inference_alg_results.update(scores)
inference_alg_results['map_cluster_assignments'] = map_cluster_assignments

wandb.log(scores, step=0)

# Additionally log the (posterior over the) number of clusters per obs
if config['inference_alg_str'] == 'RN-CRP':
    wandb.log({'num_clusters_posteriors': inference_alg_results['num_clusters_posteriors']},
              step=0)

data_to_store = dict(
    config=dict(config),  # Need to convert WandB config to proper dict
    inference_alg_results=inference_alg_results,
    scores=scores)

joblib.dump(data_to_store,
            filename=inference_alg_results_path)

# rncrp.plot.plot_inference_results(
#     sampled_mog_data=sampled_mog_data,
#     inference_results=stored_data['inference_results'],
#     inference_alg_str=stored_data['inference_alg_str'],
#     inference_alg_param=stored_data['inference_alg_params'],
#     plot_dir=inference_results_dir)


print('Finished run.')
