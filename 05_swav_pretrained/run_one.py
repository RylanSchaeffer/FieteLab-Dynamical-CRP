"""
Perform inference on SwAV ImageNet embeddings for the specified inference
algorithm and model parameters.

Example usage:

05_swav_pretrained/run_one.py
"""

import joblib
import numpy as np
import os
import wandb


import rncrp.data.real_nontabular
import rncrp.helpers.dynamics
import rncrp.helpers.run
import rncrp.metrics

config_defaults = {
    'inference_alg_str': 'Dynamical-CRP',
    'dynamics_str': 'step',
    'dynamics_a': 1.,
    'dynamics_b': 0.,
    'alpha': 1.1,
    'beta': 0.,
    'likelihood_kappa': 5.,
    'n_samples': 10,
    'repeat_idx': 0,
    'vi_param_initialization': 'observation',
    'which_prior_prob': 'DP',
    'update_new_cluster_parameters': False,
    'robbins_monro_cavi_updates': True,
    'imagenet_split': 'val',
}

wandb.init(project='dcrp-swav-pretrained',
           config=config_defaults)
config = wandb.config

print(f'Running:')
for key, value in config.items():
    print(key, ' : ', value)

# determine paths
exp_dir = '05_swav_pretrained'
results_dir_path = os.path.join(exp_dir, 'results')
os.makedirs(results_dir_path, exist_ok=True)
inf_alg_results_path = os.path.join(results_dir_path,
                                    f'id={wandb.run.id}.joblib')
wandb.log({'inf_alg_results_path': inf_alg_results_path},
          step=0)

# set seeds
rncrp.helpers.run.set_seed(seed=config['repeat_idx'])

swav_imagenet_dataloader = rncrp.data.real_nontabular.load_dataloader_swav_imagenet_2021(
    split=config['imagenet_split'],
    n_samples=config['n_samples'])

# Construct observation times
num_obs = len(swav_imagenet_dataloader)
observation_times = np.arange(num_obs)

# Compute true cluster assignments
true_cluster_assignments = np.zeros(shape=num_obs)
for batch_idx, batch_tensors in enumerate(swav_imagenet_dataloader):
    true_cluster_assignments[batch_idx] = batch_tensors['target'].item()

# Compute number of true clusters
wandb.log({'n_clusters': len(np.unique(true_cluster_assignments))}, step=0)

gen_model_params = {
    'mixing_params': {
        'alpha': config['alpha'],
        'beta': config['beta'],
        'dynamics_str': config['dynamics_str'],
        'dynamics_params': {'a': config['dynamics_a'], 'b': 0.},
    },
    'component_prior_params': {
        # 'centroids_prior_cov_prefactor': config['centroids_prior_cov_prefactor']
    },
    'likelihood_params': {
        'distribution': 'vonmises_fisher',
        'likelihood_kappa': config['likelihood_kappa']
    }
}

inference_alg_kwargs = dict(
    vi_param_initialization=config['vi_param_initialization'],
    which_prior_prob=config['which_prior_prob'],
    update_new_cluster_parameters=config['update_new_cluster_parameters'],
    robbins_monro_cavi_updates=config['robbins_monro_cavi_updates'],
)

inference_alg_results = rncrp.helpers.run.run_inference_alg(
    inference_alg_str=config['inference_alg_str'],
    observations=swav_imagenet_dataloader,
    observations_times=observation_times,
    gen_model_params=gen_model_params,
    inference_alg_kwargs=inference_alg_kwargs,
)

scores, map_cluster_assignments = rncrp.metrics.compute_predicted_clusters_scores(
    cluster_assignment_posteriors=inference_alg_results['cluster_assignment_posteriors'],
    true_cluster_assignments=true_cluster_assignments,
)
inference_alg_results.update(scores)
inference_alg_results['map_cluster_assignments'] = map_cluster_assignments
wandb.log(scores, step=0)


cluster_multiclass_classification_score = rncrp.metrics.compute_cluster_multiclass_classification_score(
    cluster_assignment_posteriors=inference_alg_results['cluster_assignment_posteriors'],
    targets=true_cluster_assignments)
wandb.log({
    'avg_finetune_acc': cluster_multiclass_classification_score['avg_acc']},
    step=0)


data_to_store = dict(
    config=dict(config),  # Need to convert WandB config to proper dict
    inference_alg_results=inference_alg_results,
    true_cluster_assignments=true_cluster_assignments,  # Save because otherwise onerous to regenerate
    scores=scores)

joblib.dump(data_to_store,
            filename=inf_alg_results_path)

print('Finished run.')
