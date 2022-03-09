"""
Perform inference in a synthetic mixture of von Mises-Fisher for the specified inference
algorithm and model parameters.

Example usage:

03_mixture_of_vonmises_fisher/run_one.py
"""

import joblib
import numpy as np
import os
import wandb


import rncrp.data.synthetic
import rncrp.helpers.dynamics
import rncrp.helpers.run
import rncrp.metrics
import rncrp.plot.plot_general

config_defaults = {
    'inference_alg_str': 'Dynamical-CRP',
    'dynamics_str': 'step',
    # 'dynamics_a': 1.,
    # 'dynamics_b': 1.,
    # 'dynamics_c': 1.,
    # 'dynamics_omega': np.pi / 2.,
    'n_samples': 1000,
    'n_features': 10,
    'alpha': 5.,
    'beta': 0.,
    'likelihood_kappa': 5.,
    'repeat_idx': 0,
}

wandb.init(project='dcrp-mixture-of-vonmises-fisher',
           config=config_defaults)
config = wandb.config

print(f'Running:')
for key, value in config.items():
    print(key, ' : ', value)

# determine paths
exp_dir = '03_mixture_of_vonmises_fisher'
results_dir_path = os.path.join(exp_dir, 'results')
os.makedirs(results_dir_path, exist_ok=True)
inf_alg_results_path = os.path.join(results_dir_path,
                                    f'id={wandb.run.id}.joblib')
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
    component_prior_str='vonmises_fisher',
    component_prior_params={'likelihood_kappa': config['likelihood_kappa']})

wandb.log(
    {'n_clusters': len(np.unique(mixture_model_results['cluster_assignments']))},
    step=0)

gen_model_params = {
    'mixing_params': {
        'alpha': config['alpha'],
        'beta': config['beta'],
        'dynamics_str': config['dynamics_str'],
        'dynamics_params': mixture_model_results['dynamics_params']
    },
    'component_prior_params': {
    },
    'likelihood_params': {
        'distribution': 'vonmises_fisher',
        'likelihood_kappa': config['likelihood_kappa']
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

sum_sqrd_distances = rncrp.metrics.compute_sum_of_squared_distances_to_nearest_center(
    X=mixture_model_results['observations'],
    centroids=inference_alg_results['inference_alg'].centroids_after_last_obs())
inference_alg_results['training_reconstruction_error'] = sum_sqrd_distances
wandb.log({'training_reconstruction_error': sum_sqrd_distances}, step=0)


data_to_store = dict(
    config=dict(config),  # Need to convert WandB config to proper dict
    inference_alg_results=inference_alg_results,
    mixture_model_results=mixture_model_results,
    scores=scores)

joblib.dump(data_to_store,
            filename=inf_alg_results_path)

inf_alg_plot_dir_name = ""
for key, value in dict(config).items():
    inf_alg_plot_dir_name += f"{key}={value}_"
inf_alg_plot_dir_path = os.path.join(results_dir_path, inf_alg_plot_dir_name)
os.makedirs(inf_alg_plot_dir_path, exist_ok=True)

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

print(f'Finished 03_mixture_of_vonmises_fisher/run_one.py for sweep={wandb.run.id}.')

