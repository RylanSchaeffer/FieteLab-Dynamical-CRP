"""
Perform inference in a synthetic mixture of Gaussians for the specified inference
algorithm and model parameters.

Example usage:

01_mixture_of_gaussians/run_one.py
"""

from collections import defaultdict
import joblib
import logging
import numpy as np
import os
import torch
import wandb


import rncrp.data.synthetic
import rncrp.helpers.dynamics
import rncrp.helpers.run
import rncrp.metrics
import rncrp.plot.plot_general


config_defaults = {
    'inference_alg_str': 'CollapsedGibbsSampler',
    # 'inference_alg_str': 'DP-Means (Offline)',
    # 'inference_alg_str': 'Dynamical-CRP',
    # 'inference_alg_str': 'Dynamical-CRP (Cutoff=1e-3)',
    # 'inference_alg_str': 'Recursive-CRP',
    # 'inference_alg_str': 'K-Means (Offline)',
    # 'inference_alg_str': 'K-Means (Online)',
    # 'inference_alg_str': 'VI-GMM',
    'dynamics_str': 'step',
    'dynamics_a': 1.,
    'dynamics_b': 0.5,
    'dynamics_c': 0.5,
    'dynamics_omega': np.pi / 2.,
    'n_samples': 1000,
    'n_features': 15,
    # 'alpha': 1.1,
    # 'alpha': 4.5,
    'alpha': 5,
    'beta': 0.,
    'centroids_prior_cov_prefactor': 250.,
    'likelihood_cov_prefactor': 5.,
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
inf_alg_results_path = os.path.join(
    results_dir_path, f'id={wandb.run.id}.joblib')

wandb.log({'inf_alg_results_path': inf_alg_results_path},
          step=0)

# set seeds
rncrp.helpers.run.set_seed(seed=config['repeat_idx'])

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

n_clusters = len(np.unique(mixture_model_results['cluster_assignments']))
wandb.log(
    {'n_clusters': n_clusters},
    step=0)

gen_model_params = {
    'mixing_params': {
        'alpha': config['alpha'],
        'beta': config['beta'],
        'dynamics_str': config['dynamics_str'],
        'dynamics_params': mixture_model_results['dynamics_params']
    },
    'component_prior_params': {
        'centroids_prior_cov_prefactor': config['centroids_prior_cov_prefactor']
    },
    'likelihood_params': {
        'distribution': 'multivariate_normal',
        'likelihood_cov_prefactor': config['likelihood_cov_prefactor']
    }
}

# K-Means gets access to ground-truth number of clusters
inference_alg_kwargs = dict()
if config['inference_alg_str'].startswith('K-Means'):
    inference_alg_kwargs['n_clusters'] = n_clusters


inference_alg_results = rncrp.helpers.run.run_inference_alg(
    inference_alg_str=config['inference_alg_str'],
    observations=mixture_model_results['observations'],
    observations_times=mixture_model_results['observations_times'],
    gen_model_params=gen_model_params,
    inference_alg_kwargs=inference_alg_kwargs,
)

if config['inference_alg_str'] == 'CollapsedGibbsSampler':

    # Score each MCMC sample separately.
    num_mcmc_samples = inference_alg_results['cluster_assignments_mcmc_samples'].shape[0]
    scores_per_mcmc_sample = defaultdict(list)
    map_cluster_assignments_per_mcmc_sample = []
    for sample_idx in range(num_mcmc_samples):

        # Score the MCMC sample.
        mcmc_sample_scores, mcmc_sample_map_cluster_assignments = rncrp.metrics.compute_predicted_clusters_scores(
            cluster_assignment_posteriors=inference_alg_results['cluster_assignments_mcmc_samples'][sample_idx],
            true_cluster_assignments=mixture_model_results['cluster_assignments'])

        # Store results to be aggregated.
        for score, score_val in mcmc_sample_scores.items():
            scores_per_mcmc_sample[score].append(score_val)
        map_cluster_assignments_per_mcmc_sample.append(mcmc_sample_map_cluster_assignments)

    # Average scores over MCMC samples
    scores = dict()
    for score, score_list in scores_per_mcmc_sample.items():
        scores[score] = np.mean(score_list)

    # TODO: These values don't make sense. This averages over cluster IDs.
    inference_alg_results['map_cluster_assignments_mcmc_samples'] = np.stack(map_cluster_assignments_per_mcmc_sample)

else:
    scores, map_cluster_assignments = rncrp.metrics.compute_predicted_clusters_scores(
        cluster_assignment_posteriors=inference_alg_results['cluster_assignment_posteriors'],
        true_cluster_assignments=mixture_model_results['cluster_assignments'])
    inference_alg_results['map_cluster_assignments'] = map_cluster_assignments
inference_alg_results.update(scores)
wandb.log(scores, step=0)

# sum_sqrd_distances = rncrp.metrics.compute_sum_of_squared_distances_to_nearest_center(
#     X=mixture_model_results['observations'],
#     centroids=inference_alg_results['inference_alg'].centroids_after_last_obs())
# inference_alg_results['training_reconstruction_error'] = sum_sqrd_distances
# wandb.log({'training_reconstruction_error': sum_sqrd_distances}, step=0)


data_to_store = dict(
    config=dict(config),  # Need to convert WandB config to proper dict
    inference_alg_results=inference_alg_results,
    mixture_model_results=mixture_model_results,
    scores=scores)

joblib.dump(data_to_store,
            filename=inf_alg_results_path)


inf_alg_plot_dir_name = ""
for key, value in dict(config).items():
    # Need to truncate file names because too long
    if key in {'beta', 'vi_param_initialization', 'observation_which_prior_prob',
               'update_new_cluster_parameters', 'robbins_monro_cavi_updates'}:
        continue
    if key == 'centroids_prior_cov_prefactor':
        key = 'centroids_prior_cov'
    if key == 'likelihood_cov_prefactor':
        key = 'likelihood_cov'

    inf_alg_plot_dir_name += f"{key}={value}_"
inf_alg_plot_dir_path = os.path.join(results_dir_path, inf_alg_plot_dir_name)
os.makedirs(inf_alg_plot_dir_path, exist_ok=True)

try:
    rncrp.plot.plot_general.plot_cluster_assignments_inferred_vs_true(
        true_cluster_assignments_one_hot=mixture_model_results['cluster_assignments_one_hot'],
        cluster_assignment_posteriors=inference_alg_results['cluster_assignment_posteriors'],
        plot_dir=inf_alg_plot_dir_path,
    )

    rncrp.plot.plot_general.plot_cluster_coassignments_inferred_vs_true(
        true_cluster_assignments=mixture_model_results['cluster_assignments'],
        cluster_assignment_posteriors=inference_alg_results['cluster_assignment_posteriors'],
        plot_dir=inf_alg_plot_dir_path,
    )

# Some algorithms e.g. CollapsedGibbsSampler don't have cluster assignments.
# They will throw a KeyError; we gracefully exit instead.
except KeyError:
    pass


print(f'Finished 01_mixture_of_gaussians/run_one.py for run={wandb.run.id}.')
