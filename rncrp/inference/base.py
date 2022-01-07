import abc
# import jax
# import jax.random
import logging
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import os
import scipy.linalg
import scipy.special as sps
import scipy.stats
from scipy.stats import poisson
import torch
from typing import Dict, Tuple, Union

import rncrp.helpers.numpy_helpers
import rncrp.helpers.torch_helpers

torch.set_default_tensor_type('torch.FloatTensor')


class BaseModel(abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        self.model_str = None
        self.gen_model_params = None
        self.plot_dir = None
        self.fit_results = None

    @abc.abstractmethod
    def fit(self,
            observations: np.ndarray,
            observations_times: np.ndarray):
        pass

    # @abc.abstractmethod
    # def sample_variables_for_predictive_posterior(self,
    #                                               num_samples: int):
    #     pass
    #
    # @abc.abstractmethod
    # def features_after_last_obs(self) -> np.ndarray:
    #     """
    #     Returns array of shape (num features, feature dimension)
    #     """
    #     pass
    #
    # @abc.abstractmethod
    # def features_by_obs(self) -> np.ndarray:
    #     """
    #     Returns array of shape (num obs, num features, feature dimension)
    #     :return:
    #     """
    #     pass


