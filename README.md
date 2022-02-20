# Streaming Inference for Infinite Non-Stationary Clustering

### Authors: Rylan Schaeffer, Gabrielle Kaili-May Liu, Yilun Du, Ila Rani Fiete

-----

This code corresponds to our .


## Setup

After cloning the repository, create a virtual environment for Python 3:

`python3 -m venv dcrp_venv`

Then activate the virtual environment:

`source dcrp_venv/bin/activate`

Ensure pip is up to date:

`pip install --upgrade pip`

Then install the required packages:

`pip install -r requirements.txt`

We did not test Python2, but Python2 may work.


## Running

Each experiment (e.g. `00_prior`) is in its own directory. Each experiment directory should contain a 
`run_one.py` file for running a single configuration. If you want to launch sweeps, 
we run sweeps via [Weights and Biases](https://wandb.ai/) that we configure via `.yaml` files. 
After completing your runs, each experiment directory should also contain an `analyze_sweep.py`
file that reads the results from Weights and Biases and generates the plots in a `plots`
subdirectory (e.g. `00_prior/plots`).

## Contact

Questions? Comments? Interested in collaborating? Open an issue or 
email Rylan Schaeffer (rylanschaeffer@gmail.com) and cc Ila Fiete (fiete@mit.edu).
