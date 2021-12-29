"""
Launch run_one.py with each configuration of the parameters, to compare the
analytical DCRP marginal distribution against Monte-Carlo estimates of the DCRP
marginal distribution.

Example usage:

00_prior/run_all.py
"""

import itertools
import logging
import os
import subprocess


def run_all():
    # create directory
    exp_dir_path = '00_prior'
    results_dir_path = os.path.join(exp_dir_path, 'results')
    os.makedirs(results_dir_path, exist_ok=True)

    num_customers = [50]
    num_mc_samples = [5000]  # number of Monte Carlo samples to draw
    alphas = [1.1, 10.78, 15.37, 30.91]
    betas = [0.]
    dynamics_strs = ['step', 'exp', 'sinusoid', 'hyperbolic']

    hyperparams = [num_customers, num_mc_samples, alphas, betas, dynamics_strs]
    for num_customer, num_mc_sample, alpha, beta, dynamics_str in itertools.product(*hyperparams):
        launch_run_one(
            exp_dir_path=exp_dir_path,
            results_dir_path=results_dir_path,
            num_customer=num_customer,
            num_mc_sample=num_mc_sample,
            alpha=alpha,
            beta=beta,
            dynamics_str=dynamics_str)


def launch_run_one(exp_dir_path: str,
                   results_dir_path: str,
                   num_customer: int,
                   num_mc_sample: int,
                   alpha: float,
                   beta: float,
                   dynamics_str: str):

    if beta != 0:
        logging.info('Beta not implemented. Setting beta to 0.')
        beta = 0.

    run_one_script_path = os.path.join(exp_dir_path, 'run_one.sh')
    command_and_args = [
        'sbatch',
        run_one_script_path,
        results_dir_path,
        str(num_customer),
        str(num_mc_sample),
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
