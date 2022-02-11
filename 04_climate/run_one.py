"""

"""

import joblib
import numpy as np
import os
import wandb

# import plot_climate
import rncrp.data.real
import rncrp.helpers.dynamics
import rncrp.helpers.run
import rncrp.metrics

config_defaults = {
    'inference_alg_str': 'RN-CRP',
    'dynamics_str': 'step',
    'dynamics_a': 1.,
    'dynamics_b': 1.,
    'dynamics_c': 1.,
    'dynamics_omega': np.pi / 2.,
    'alpha': 5.9,
    'beta': 0.,
    'repeat_idx': 0,
}

wandb.init(project='rncrp-climate',
           config=config_defaults)
config = wandb.config

print(f'Running:')
for key, value in config.items():
    print(key, ' : ', value)

# determine paths
exp_dir = 'exp2_climate'
results_dir_path = os.path.join(exp_dir, 'results')
os.makedirs(results_dir_path, exist_ok=True)
inference_alg_results_path = os.path.join(results_dir_path,
                                          f'id={wandb.run.id}.joblib')

# set seeds
rncrp.helpers.run.set_seed(config['repeat_idx'])

climate_data_results = rncrp.data.real.load_dataset_climate()
gen_model_params = {
    'mixing_params': {
        'alpha': config['alpha'],
        'beta': config['beta'],
        # 'dynamics_str': config['dynamics_str'],
        # 'dynamics_params': mixture_model_results['dynamics_params']
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
    observations=climate_data_results['annual'], ## TODO: USE MONTHLY DATA INSTEAD?
    observations_times=None, ## TODO: FILL IN??
    gen_model_params=gen_model_params,
)

scores, map_cluster_assignments = rncrp.metrics.compute_predicted_clusters_scores(
    cluster_assignment_posteriors=inference_alg_results['cluster_assignment_posteriors'],
    true_cluster_assignments=climate_data_results['cluster_assignments'],
)

inference_alg_results.update(scores)
inference_alg_results['map_cluster_assignments'] = map_cluster_assignments

wandb.log(scores, step=0)

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
