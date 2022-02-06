"""

"""
import itertools
import joblib
import logging
import os
import subprocess

import numpy as np

import plot
import rncrp.data

def run_all():
    # create directory
    exp_dir_path = 'exp2_climate'
    results_dir_path = os.path.join(exp_dir_path, 'results')
    os.makedirs(results_dir_path, exist_ok=True)

    # cluster_assignment_samplings = [
    #     ('categorical', dict(probs=np.ones(5)/5.)),
    #     ('categorical', dict(probs=np.array([0.4, 0.25, 0.2, 0.1, 0.05]))),
    #     ('D-CRP', dict(dynamics_str='perfectintegrator', alpha=5.98, beta=0.)),
    #     ('D-CRP', dict(dynamics_str='leakyintegrator', alpha=5.98, beta=0.)),
    #     ('D-CRP', dict(dynamics_str='harmonicoscillator', alpha=5.98, beta=0.))
    # ]
    # alphas = [1.1, 10.78, 15.37, 30.91]
    alphas = np.round(np.linspace(1.1, 30.91, 20), 2)
    betas = [0.]
    # betas = [0.3, 5.6, 12.9, 21.3]
    dynamics_strs = ['perfectintegrator', 'leakyintegrator', 'harmonicoscillator']
    inference_alg_strs = ['D-CRP']
    hyperparams = [alphas, betas, inference_alg_strs, dynamics_strs]

    # load dataset
    sampled_climate_data = rncrp.data.real.load_dataset_climate()
    dataset_dir = os.path.join(results_dir_path, 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    joblib.dump(sampled_climate_data,
                filename=os.path.join(dataset_dir, 'data.joblib'))

    # launch inference
    for alpha, beta, inference_alg_str, dynamics_str in itertools.product(*hyperparams):
        launch_run_one(
            exp_dir_path=exp_dir_path,
            dataset_dir=dataset_dir,
            alpha=alpha,
            beta=beta,
            inference_alg_str=inference_alg_str,
            dynamics_str=dynamics_str)


def launch_run_one(exp_dir_path: str,
                   dataset_dir: str,
                   inference_alg_str: str,
                   alpha: float,
                   beta: float,
                   dynamics_str: str,):

    run_one_script_path = os.path.join(exp_dir_path, 'run_one.sh')
    command_and_args = [
        'sbatch',
        run_one_script_path,
        dataset_dir,
        inference_alg_str,
        str(alpha),
        str(beta),
        dynamics_str]

    # TODO: Figure out where the logger is logging to
    logging.info(f'Launching ' + ' '.join(command_and_args))
    subprocess.run(command_and_args)
    logging.info(f'Launched ' + ' '.join(command_and_args))


if __name__ == '__main__':
    run_all()
    logging.info('Finished.')