# Streaming Inference for Nonstationary Infinite Mixture Models

### Authors: Rylan Schaeffer, Gabrielle Kaili-May Liu, Ila Rani Fiete

-----

This code corresponds to our .


## Setup

After cloning the repository, create a virtual environment for Python 3:

`python3 -m venv rncrp_venv`

Then activate the virtual environment:

`source rncrp_venv/bin/activate`

Ensure pip is up to date:

`pip install --upgrade pip`

Then install the required packages:

`pip install -r requirements.txt`

We did not test Python2, but Python2 may work.


## Running

Each experiment has its own directory, each containing a `main.py` that creates a `plots`
subdirectory (e.g. `exp_00_ibp_prior/plots`) and then reproduces the plots in the paper. Each 
`main.py` should be run from the repository directory e.g.:

`python3 exp_00_ibp_prior/main.py`

## TODO
- Write out mean field family and approximate variational lower bound
- Derive parameter updates for Gaussian likelihood and probably one other (e.g. multinomial)
- Generate synthetic data with particular temporal structure - assume the algorithm knows exactly the generative process
- Real-world data: Omniglot, with particular temporal transition between "languages"
- Real-world data: NYT Corpus or some other text corpus
- Add to Unsupervised Learning of Visual Features by Contrasting Cluster Assignments

## Contact

Questions? Comments? Interested in collaborating? Open an issue or 
email Rylan Schaeffer (rylanschaeffer@gmail.com) and cc Ila Fiete (fiete@mit.edu).
