import abc
import numpy as np
import torch
from typing import Dict

from rncrp.helpers.torch_helpers import torch_round

dynamics_strs = [
    'perfectintegrator',
    'leakyintegrator',
    'harmonicoscillator',
    'hyperbolic',
    'statetransition']


class Dynamics(abc.ABC):

    def __init__(self, params):
        self.params = params
        self._state = None
        super().__init__()

    @abc.abstractmethod
    def initialize_state(self,
                         customer_assignment_probs: np.ndarray,
                         time: float, ) -> Dict[str, np.ndarray]:
        pass

    @abc.abstractmethod
    def run_dynamics(self,
                     time_start: float,
                     time_end: float
                     ) -> Dict[str, np.ndarray]:
        pass

    @abc.abstractmethod
    def update_state(self,
                     customer_assignment_probs: np.ndarray,
                     time: float,
                     ) -> Dict[str, np.ndarray]:
        pass


class LinearFirstOrderNumpy(Dynamics):

    def __init__(self,
                 params: Dict[str, float] = None):
        """
        Calculate N(t_1) from N(t_0), t_0, t_1.

        Dynamics: a dN/dt + b N = 0
        Initial conditions: N(t_0)
            Solution: N(t_1) = N(t_0) e^{- b (t_1 - t_0) / a}
        """

        if params is None:
            params = {'a': 1., 'b': 1.}
        assert 'a' in params
        assert 'b' in params
        super().__init__(params=params)

    def initialize_state(self,
                         customer_assignment_probs: np.ndarray,
                         time: float,
                         ) -> Dict[str, np.ndarray]:
        del time
        self._state = {
            'N': customer_assignment_probs}
        return self._state

    def run_dynamics(self,
                     time_start: float,
                     time_end: float) -> Dict[str, np.ndarray]:
        assert time_start < time_end
        time_delta = time_end - time_start
        exp_change = np.exp(- self.params['b'] * time_delta / self.params['a'])
        self._state['N'] = exp_change * self._state['N']
        return self._state

    def update_state(self,
                     customer_assignment_probs: np.ndarray,
                     time: float,
                     ) -> Dict[str, np.ndarray]:
        self._state['N'] += customer_assignment_probs
        return self._state


class LinearFirstOrderTorch(Dynamics):

    def __init__(self,
                 params: Dict[str, float] = None):
        """
        Calculate N(t_1) from N(t_0), t_0, t_1.

        Dynamics: a dN/dt + b N = 0
        Initial conditions: N(t_0)
            Solution: N(t_1) = N(t_0) e^{- b (t_1 - t_0) / a}
        """

        if params is None:
            params = {'a': 1., 'b': 1.}
        assert 'a' in params
        assert 'b' in params
        super().__init__(params=params)

    def initialize_state(self,
                         customer_assignment_probs: torch.Tensor,
                         time: float,
                         ) -> Dict[str, torch.Tensor]:
        del time
        self._state = {
            'N': customer_assignment_probs}
        return self._state

    def run_dynamics(self,
                     time_start: float,
                     time_end: float) -> Dict[str, torch.Tensor]:
        assert time_start < time_end
        time_delta = time_end - time_start
        exp_change = torch.exp(- self.params['b'] * time_delta / self.params['a'])
        self._state['N'] = exp_change * self._state['N']
        return self._state

    def update_state(self,
                     customer_assignment_probs: torch.Tensor,
                     time: float,
                     ) -> Dict[str, torch.Tensor]:
        self._state['N'] += customer_assignment_probs
        return self._state


class HarmonicOscillatorNumpy(Dynamics):
    """

    """

    def __init__(self,
                 params: Dict[str, float] = None):
        if params is None:
            params = {'omega': 1.}
        assert 'omega' in params
        super().__init__(params=params)

    def initialize_state(self,
                         customer_assignment_probs: np.ndarray,
                         time: float,
                         ) -> Dict[str, np.ndarray]:
        cos_coeff = 0.5 * np.multiply(
            np.cos(np.full_like(customer_assignment_probs,
                                fill_value=self.params['omega'] * time)),
            customer_assignment_probs)
        sin_coeff = 0.5 * np.multiply(
            np.sin(np.full_like(customer_assignment_probs,
                                fill_value=self.params['omega'] * time)),
            customer_assignment_probs)
        const_coeff = 0.5 * customer_assignment_probs
        self._state = {
            'cos_coeffs': cos_coeff,
            'sin_coeffs': sin_coeff,
            'const_coeffs': const_coeff,
        }
        self._add_N_to_state(time=time)
        return self._state

    def run_dynamics(self,
                     time_start: float,
                     time_end: float
                     ) -> Dict[str, np.ndarray]:
        self._add_N_to_state(time=time_end)
        return self._state

    # def run_dynamics(self, t_start, t_end) -> Dict[str, np.ndarray]:
    #     assert t_start < t_end
    #     # solve [N(t_0), N'(0)]^T = [[1, 1], [s_1, s_2]] [c_1, c_2]
    #     init_conditions = np.array([self._state['N'], self._state['dN/dt']])
    #     # shape (2, max number of tables)
    #     coefficients = np.matmul(self._coefficient_matrix, init_conditions)
    #     N_end = np.add(
    #         coefficients[0, :] * np.exp(self._s1 * t_end),
    #         coefficients[1, :] * np.exp(self._s2 * t_end))
    #
    #     dNdt_end = np.add(
    #         coefficients[0, :] * self._s1 * np.exp(self._s1 * t_end),
    #         coefficients[1, :] * self._s2 * np.exp(self._s2 * t_end))
    #
    #     self._state = {
    #         'N': N_end.real,
    #         'dN/dt': dNdt_end.real,
    #     }
    #
    #     # coordinate transform so that each customer oscillates between 0 and 1
    #     # rather than between -1. and 1. Also add mask so that + 0.5 only affects
    #     # occupied tables
    #     transformed_state = {
    #         'N': np.multiply(0.5 + 0.5 * self._state['N'].copy(),
    #                          self._state['N'].real != 0)
    #     }
    #
    #     return transformed_state

    def update_state(self,
                     customer_assignment_probs: np.ndarray,
                     time: float,
                     ) -> Dict[str, np.ndarray]:
        new_cos_coeff = 0.5 * np.multiply(
            np.cos(self.params['omega'] * time),
            customer_assignment_probs)
        new_sin_coeff = 0.5 * np.multiply(
            np.sin(self.params['omega'] * time),
            customer_assignment_probs)
        new_const_coeff = 0.5 * customer_assignment_probs
        self._state['cos_coeffs'] += new_cos_coeff
        self._state['sin_coeffs'] += new_sin_coeff
        self._state['const_coeffs'] += new_const_coeff
        self._add_N_to_state(time=time)
        return self._state

    def _add_N_to_state(self, time):
        N = self._state['const_coeffs'].copy()
        N += self._state['cos_coeffs'].copy() * np.cos(self.params['omega'] * time)
        N += self._state['sin_coeffs'].copy() * np.sin(self.params['omega'] * time)

        # sometimes, floating point errors will give N values like -9.18e-17
        # This will break the code if we use these values to sample from a Categorical,
        # so we need to round
        N = np.round(N, decimals=12)
        self._state['N'] = N


class HarmonicOscillatorTorch(Dynamics):
    """

    """

    def __init__(self,
                 params: Dict[str, float] = None):
        if params is None:
            params = {'omega': 1.}
        assert 'omega' in params
        super().__init__(params=params)

    def initialize_state(self,
                         customer_assignment_probs: torch.Tensor,
                         time: float,
                         ) -> Dict[str, torch.Tensor]:
        cos_coeff = 0.5 * torch.multiply(
            torch.cos(torch.full_like(customer_assignment_probs,
                                      fill_value=self.params['omega'] * time)),
            customer_assignment_probs)
        sin_coeff = 0.5 * torch.multiply(
            torch.sin(torch.full_like(customer_assignment_probs,
                                      fill_value=self.params['omega'] * time)),
            customer_assignment_probs)
        const_coeff = 0.5 * customer_assignment_probs
        self._state = {
            'cos_coeffs': cos_coeff,
            'sin_coeffs': sin_coeff,
            'const_coeffs': const_coeff,
        }
        self._add_N_to_state(time=time)
        return self._state

    def run_dynamics(self,
                     time_start: float,
                     time_end: float
                     ) -> Dict[str, torch.Tensor]:
        self._add_N_to_state(time=time_end)
        return self._state

    # def run_dynamics(self, t_start, t_end) -> Dict[str, np.ndarray]:
    #     assert t_start < t_end
    #     # solve [N(t_0), N'(0)]^T = [[1, 1], [s_1, s_2]] [c_1, c_2]
    #     init_conditions = np.array([self._state['N'], self._state['dN/dt']])
    #     # shape (2, max number of tables)
    #     coefficients = np.matmul(self._coefficient_matrix, init_conditions)
    #     N_end = np.add(
    #         coefficients[0, :] * np.exp(self._s1 * t_end),
    #         coefficients[1, :] * np.exp(self._s2 * t_end))
    #
    #     dNdt_end = np.add(
    #         coefficients[0, :] * self._s1 * np.exp(self._s1 * t_end),
    #         coefficients[1, :] * self._s2 * np.exp(self._s2 * t_end))
    #
    #     self._state = {
    #         'N': N_end.real,
    #         'dN/dt': dNdt_end.real,
    #     }
    #
    #     # coordinate transform so that each customer oscillates between 0 and 1
    #     # rather than between -1. and 1. Also add mask so that + 0.5 only affects
    #     # occupied tables
    #     transformed_state = {
    #         'N': np.multiply(0.5 + 0.5 * self._state['N'].copy(),
    #                          self._state['N'].real != 0)
    #     }
    #
    #     return transformed_state

    def update_state(self,
                     customer_assignment_probs: torch.Tensor,
                     time: torch.Tensor,
                     ) -> Dict[str, torch.Tensor]:
        new_cos_coeff = 0.5 * torch.multiply(
            torch.cos(self.params['omega'] * time),
            customer_assignment_probs)
        new_sin_coeff = 0.5 * torch.multiply(
            torch.sin(self.params['omega'] * time),
            customer_assignment_probs)
        new_const_coeff = 0.5 * customer_assignment_probs
        self._state['cos_coeffs'] += new_cos_coeff
        self._state['sin_coeffs'] += new_sin_coeff
        self._state['const_coeffs'] += new_const_coeff
        self._add_N_to_state(time=time)
        return self._state

    def _add_N_to_state(self,
                        time: torch.Tensor):
        N = self._state['const_coeffs']
        N += self._state['cos_coeffs'] * torch.cos(self.params['omega'] * time)
        N += self._state['sin_coeffs'] * torch.sin(self.params['omega'] * time)

        # sometimes, floating point errors will give N values like -9.18e-17
        # This will break the code if we use these values to sample from a Categorical,
        # so we need to round
        N = torch_round(N, decimals=6)
        self._state['N'] = N


class HyperbolicNumpy(Dynamics):

    def __init__(self,
                 params: Dict[str, float] = None):
        """
        Creates D(\Delta) = \frac{1}{1 + c\Delta}
        """

        if params is None:
            params = {'c': 1., 'num_exponentials': 250}
        if 'num_exponentials' not in params:
            params['num_exponentials'] = 250
        assert 'c' in params
        super().__init__(params=params)

        # We approximate the integral with a Riemann sum.
        # Shift by a half-width to use the midpoint.
        width = 0.05
        self._exponential_rates = np.linspace(start=0,
                                              stop=width * params['num_exponentials'],
                                              num=params['num_exponentials']) + width / 2

        self._probabilities = width * np.exp(-self._exponential_rates / params['c']) / params['c']

        # Add a trailing dimension to make addition / multiplication easy
        self._exponential_rates = self._exponential_rates[:, np.newaxis]

    def initialize_state(self,
                         customer_assignment_probs: np.ndarray,
                         time: float,
                         ) -> Dict[str, np.ndarray]:

        exponential_Ns = np.repeat(customer_assignment_probs[np.newaxis, :],
                                   repeats=self.params['num_exponentials'],
                                   axis=0)

        N_weighted_avg = np.matmul(self._probabilities, exponential_Ns)
        self._state = {
            'N': N_weighted_avg,
            'exponential_Ns': exponential_Ns}
        return self._state

    def run_dynamics(self,
                     time_start: float,
                     time_end: float) -> Dict[str, np.ndarray]:
        assert time_start < time_end
        exp_change = np.exp(- self._exponential_rates * (time_end - time_start))
        self._state['exponential_Ns'] *= exp_change
        self._state['N'] = np.matmul(self._probabilities,
                                     self._state['exponential_Ns'])
        return self._state

    def update_state(self,
                     customer_assignment_probs: np.ndarray,
                     time: float,
                     ) -> Dict[str, np.ndarray]:

        self._state['exponential_Ns'] += customer_assignment_probs[np.newaxis, :]
        self._state['N'] = np.matmul(self._probabilities,
                                     self._state['exponential_Ns'])
        return self._state


class HyperbolicTorch(Dynamics):

    def __init__(self,
                 params: Dict[str, float] = None):
        """
        Creates D(\Delta) = \frac{1}{1 + c\Delta}
        """

        if params is None:
            params = {'c': 1., 'num_exponentials': 250}
        if 'num_exponentials' not in params:
            params['num_exponentials'] = 250
        assert 'c' in params
        super().__init__(params=params)

        # We approximate the integral with a Riemann sum.
        # Shift by a half-width to use the midpoint.
        width = 0.05
        self._exponential_rates = torch.linspace(start=0,
                                                 end=width * params['num_exponentials'],
                                                 steps=params['num_exponentials']) + width / 2

        self._weights = width * torch.exp(-self._exponential_rates / params['c']) / params['c']

        # Add a trailing dimension to make addition / multiplication easy
        self._exponential_rates = self._exponential_rates[:, np.newaxis]

    def initialize_state(self,
                         customer_assignment_probs: torch.Tensor,
                         time: torch.Tensor,
                         ) -> Dict[str, torch.Tensor]:

        # TODO: convert this to pytorch
        # Shape: (num exponentials, max num clusters)
        exponential_Ns = customer_assignment_probs[np.newaxis, :].repeat(
            self.params['num_exponentials'], 1)

        N_weighted_avg = torch.matmul(self._weights, exponential_Ns)
        self._state = {
            'N': N_weighted_avg,
            'exponential_Ns': exponential_Ns}
        return self._state

    def run_dynamics(self,
                     time_start: torch.Tensor,
                     time_end: torch.Tensor,
                     ) -> Dict[str, torch.Tensor]:
        assert time_start < time_end
        exp_change = torch.exp(- self._exponential_rates * (time_end - time_start))
        self._state['exponential_Ns'] *= exp_change
        self._state['N'] = torch.matmul(self._weights,
                                        self._state['exponential_Ns'])
        return self._state

    def update_state(self,
                     customer_assignment_probs: np.ndarray,
                     time: float,
                     ) -> Dict[str, np.ndarray]:

        self._state['exponential_Ns'] += customer_assignment_probs[np.newaxis, :]
        self._state['N'] = np.matmul(self._weights,
                                     self._state['exponential_Ns'])
        return self._state


# class LinearSecondOrder(Dynamics):
#     """
#     Calculate N(t_1) from N(t_0), t_0, t_1.
#
#     Dynamics: a d^2N/dt^2 + b dN/dt + cN = 0
#     Initial conditions: N(t_0), dN(t_0)/dt
#     Solution: N(t_1) = N(t_0) e^{- b (t_1 - t_0) / a}
#     """
#     def __init__(self,
#                  params: Dict[str, float] = None):
#
#         if params is None:
#             params = {'a': 1., 'b': 0., 'c': 1.}
#         assert 'a' in params
#         assert 'b' in params
#         assert 'c' in params
#         super().__init__(params=params)
#
#         # compute roots of characteristic polynomial
#         s1, s2 = np.roots([params['a'], params['b'], params['c']])
#         if s1 == s2:
#             raise NotImplementedError
#         self._s1, self._s2 = s1, s2
#         self._coefficient_matrix = np.linalg.inv(np.array([[1., 1.], [s1, s2]]))
#
#     def initialize_state(self,
#                          customer_assignment_probs: np.ndarray,
#                          t: float) -> Dict[str, np.ndarray]:
#         # treat first observation as occuring at time zero
#         cos_coeff = np.multiply(
#             0.5 * np.cos(np.full_like(customer_assignment_probs, fill_value=t)),
#             customer_assignment_probs)
#         sin_coeff = np.zeros_like(customer_assignment_probs)
#         const_coeff = customer_assignment_probs / 2
#         self._state = {
#             'N': const_coeff + cos_coeff + sin_coeff,
#             'cos_coeff': cos_coeff,
#             'sin_coeff': sin_coeff,
#             'const_coeff': const_coeff,
#         }
#         return self._state
#
#     def run_dynamics(self, t_start, t_end) -> Dict[str, np.ndarray]:
#         new_state = {
#             'N': self._state['const_coeff']
#                  + self._state['cos_coeff'] * np.cos(t_end)
#                  + self._state['sin_coeff'] * np.sin(t_end),
#             'cos_coeff': self._state['cos_coeff'],
#             'sin_coeff': self._state['sin_coeff'],
#             'const_coeff': self._state['const_coeff'],
#         }
#         self._state = new_state
#         return self._state
#
#     # def run_dynamics(self, t_start, t_end) -> Dict[str, np.ndarray]:
#     #     assert t_start < t_end
#     #     # solve [N(t_0), N'(0)]^T = [[1, 1], [s_1, s_2]] [c_1, c_2]
#     #     init_conditions = np.array([self._state['N'], self._state['dN/dt']])
#     #     # shape (2, max number of tables)
#     #     coefficients = np.matmul(self._coefficient_matrix, init_conditions)
#     #     N_end = np.add(
#     #         coefficients[0, :] * np.exp(self._s1 * t_end),
#     #         coefficients[1, :] * np.exp(self._s2 * t_end))
#     #
#     #     dNdt_end = np.add(
#     #         coefficients[0, :] * self._s1 * np.exp(self._s1 * t_end),
#     #         coefficients[1, :] * self._s2 * np.exp(self._s2 * t_end))
#     #
#     #     self._state = {
#     #         'N': N_end.real,
#     #         'dN/dt': dNdt_end.real,
#     #     }
#     #
#     #     # coordinate transform so that each customer oscillates between 0 and 1
#     #     # rather than between -1. and 1. Also add mask so that + 0.5 only affects
#     #     # occupied tables
#     #     transformed_state = {
#     #         'N': np.multiply(0.5 + 0.5 * self._state['N'].copy(),
#     #                          self._state['N'].real != 0)
#     #     }
#     #
#     #     return transformed_state
#
#     def update_state(self, customer_assignment_probs) -> Dict[str, np.ndarray]:
#         self._state['N'] += customer_assignment_probs
#         self._state['dN/dt'] += 0.
#         return self._state


# class StateTransition(Dynamics):
#
#     def __init__(self, params: Dict[str, float] = None):
#         if params is None:
#             params = {}
#         assert len(params) == 0
#         super().__init__(params=params)
#         self._prev_customer_assignment_probs = None
#         self._new_state_idx = 0
#
#     def initialize_state(self,
#                          customer_assignment_probs: np.ndarray,
#                          time: float) -> Dict[str, np.ndarray]:
#         self._new_state_idx = 0
#         self._prev_customer_assignment_probs = customer_assignment_probs
#         unnormalized_transition = np.zeros(
#             shape=(len(customer_assignment_probs),
#                    len(customer_assignment_probs)))
#         unnormalized_transition[self._new_state_idx, self._new_state_idx] = 1.
#         self._state = {
#             'N': customer_assignment_probs.copy(),
#             'unnormalized_transition': unnormalized_transition}
#         return self._state
#
#     def run_dynamics(self,
#                      time_start: float,
#                      time_end: float) -> Dict[str, np.ndarray]:
#         normalized_transition = self._normalize_transition()
#         new_N = np.matmul(normalized_transition, self._state['N'])
#         self._state = {
#             'N': new_N,
#             'unnormalized_transition': self._state['unnormalized_transition'],
#         }
#         return self._state
#
#     def update_state(self,
#                      customer_assignment_probs: np.ndarray,
#                      time: float
#                      ) -> Dict[str, np.ndarray]:
#         self._new_state_idx += 1
#
#         # add transition probability
#         new_unnormalized_transition = np.add(
#             self._state['unnormalized_transition'],
#             np.outer(customer_assignment_probs,
#                      self._prev_customer_assignment_probs).T)
#
#         # add probability of staying in state
#         # new_unnormalized_transition[self._new_state_idx, self._new_state_idx] += 1
#
#         # replace prev customer assignment probs
#         self._prev_customer_assignment_probs = customer_assignment_probs
#         self._state = {
#             'N': self._state['N'],
#             'unnormalized_transition': new_unnormalized_transition,
#         }
#         return self._state
#
#     def _normalize_transition(self) -> np.ndarray:
#         unnormalized_transition = self._state['unnormalized_transition'].copy()
#         # we want each column to sum to 1
#         normalized_transition = np.divide(
#             unnormalized_transition,
#             np.sum(unnormalized_transition, axis=0))
#         # NaNs can result if column sum is 0; set these to 0
#         normalized_transition[np.isnan(normalized_transition)] = 0
#         return normalized_transition


def convert_dynamics_str_to_dynamics_obj(dynamics_str: str,
                                         dynamics_params: Dict[str, float] = None,
                                         implementation_mode: str = 'numpy',
                                         ) -> Dynamics:
    assert implementation_mode in {'numpy', 'torch'}

    if dynamics_str == 'step':

        if dynamics_params is None:
            dynamics_params = {'a': 1., 'b': 0.}

        if 'a' not in dynamics_params:
            dynamics_params['a'] = 1.
        if 'b' not in dynamics_params:
            dynamics_params['b'] = 0.

        assert dynamics_params['b'] == 0.

        if implementation_mode == 'numpy':
            dynamics_fn = LinearFirstOrderNumpy
        elif implementation_mode == 'torch':
            dynamics_fn = LinearFirstOrderTorch
        else:
            raise NotImplementedError

    elif dynamics_str == 'exp':
        if dynamics_params is None:
            dynamics_params = {'a': 1., 'b': 1.}

        if implementation_mode == 'numpy':
            dynamics_fn = LinearFirstOrderNumpy
        elif implementation_mode == 'torch':
            dynamics_fn = LinearFirstOrderTorch
        else:
            raise NotImplementedError

    elif dynamics_str == 'sinusoid':
        if dynamics_params is None:
            dynamics_params = {'omega': np.pi / 2}
        # dynamics = utils.dynamics.LinearSecondOrder(
        #     params={'a': 1, 'b': 0., 'c': 1.})

        if implementation_mode == 'numpy':
            dynamics_fn = HarmonicOscillatorNumpy
        elif implementation_mode == 'torch':
            dynamics_fn = HarmonicOscillatorTorch
        else:
            raise NotImplementedError

    elif dynamics_str == 'hyperbolic':
        if dynamics_params is None:
            dynamics_params = {'c': 1.}

        if implementation_mode == 'numpy':
            dynamics_fn = HyperbolicNumpy
        elif implementation_mode == 'torch':
            dynamics_fn = HyperbolicTorch
        else:
            raise NotImplementedError

    elif dynamics_str == 'statetransition':
        # dynamics = StateTransition(
        #     params={})
        raise NotImplementedError(dynamics_str)
    elif dynamics_str == 'logistic':
        # dynamics_constructor: dy/dt = a y - b y^2 + delta
        # def dynamics_fn(state: np.ndarray, time_delta: float,
        #                 a: float = 1., b: float = 1.):
        #     d = a / 1. - b
        #     return np.multiply(state, 1. / (1. + np.exp(time_delta)))
        raise NotImplementedError(dynamics_str)
    else:
        raise ValueError(f'Impermissible dynamics_str: {dynamics_str}')

    dynamics = dynamics_fn(dynamics_params)

    return dynamics


def dynamics_factory(dynamics_str: str):
    if dynamics_str == 'step':
        dynamics = LinearFirstOrderNumpy(params={'a': 1., 'b': 0.})
    elif dynamics_str == 'exp':
        dynamics = LinearFirstOrderNumpy(params={'a': 1., 'b': 1.})
    elif dynamics_str == 'sinusoid':
        dynamics = HarmonicOscillatorNumpy(params={'omega': 1.})
    else:
        raise NotImplementedError
    return dynamics
