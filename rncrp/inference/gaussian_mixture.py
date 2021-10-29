import abc
import jax
import jax.random
import logging
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import os
import scipy.linalg
from scipy.stats import poisson
import torch
from typing import Dict, Tuple, Union

import rncrp.helpers.numpy
import rncrp.helpers.torch

torch.set_default_tensor_type('torch.FloatTensor')

