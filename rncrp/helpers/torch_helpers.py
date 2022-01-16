import logging

import numpy as np
import torch


def assert_torch_no_nan_no_inf_is_real(x):
    try:
        assert torch.all(~torch.isnan(x))
    except AssertionError:
        print(x)
        raise ValueError('Contains NaNS')
    try:
        assert torch.all(~torch.isinf(x))
    except AssertionError:
        print(x)
        raise ValueError('Contains Infs')
    try:
        assert torch.all(torch.isreal(x))
    except AssertionError:
        print(x)
        raise ValueError('Contains imaginary values')


def convert_half_cov_to_cov(half_cov: torch.Tensor) -> torch.Tensor:
    """
    Converts half-covariance M into covariance M^T M in a batched manner.
    Convert batch half-covariance to covariance.

    Assumes M is [batch size, D, D] tensor. If no batch dimension exists,
    one will be added and then subsequently removed.
    """

    # if no batch dimension is given, add one
    has_batch_dim = len(half_cov.shape) == 3
    if not has_batch_dim:
        half_cov = torch.unsqueeze(half_cov, dim=0)

    cov = torch.einsum(
        'abc, abd->acd',
        half_cov,
        half_cov)
    # batch_size = half_cov.shape[0]
    # cov_test = torch.stack([torch.matmul(half_cov[k].T, half_cov[k])
    #                         for k in range(batch_size)])
    # assert torch.allclose(cov, cov_test)

    # if no batch dimension was originally given, remove
    if not has_batch_dim:
        cov = torch.squeeze(cov, dim=0)

    assert_torch_no_nan_no_inf_is_real(cov)
    return cov


def expected_log_bernoulli_under_bernoulli(p_prob: torch.Tensor,
                                           q_prob: torch.Tensor) -> torch.Tensor:
    """
    Compute E_{q(x)}[log p(x)] where p(x) and q(x)

    All inputs are expected to have a batch dimension, then appropriate shape dimensions
    i.e. if p_probs is (batch, 3), then q_probs should be (batch, 3).
    """
    assert p_prob.shape == q_prob.shape

    term_one = torch.multiply(q_prob, torch.log(p_prob))
    term_two = torch.multiply(1. - q_prob, torch.log(1. - p_prob))

    # Recall: 0 log 0 should be 0, but numerically, torch sets to NaN.
    # Consequently, we mask these terms.
    term_one[q_prob == 0.] = 0.
    term_two[q_prob == 1.] = 0.

    total = torch.sum(torch.add(term_one, term_two))
    assert_torch_no_nan_no_inf_is_real(total)
    return total


def expected_log_gaussian_under_gaussian(p_mean: torch.Tensor,
                                         p_cov: torch.Tensor,
                                         q_mean: torch.Tensor,
                                         q_cov: torch.Tensor,
                                         check_einsums: bool = False) -> torch.Tensor:
    """
    Compute E_{q(x)}[log p(x)] where p(x) and q(x) are both Gaussian.

    All inputs are expected to have a batch dimension, then appropriate shape dimensions
    i.e. if p_mean is (batch, 3), then q_mean should be (batch, 3) and p_cov and q_cov
    should both be (batch, 3, 3).
    """

    # gaussian_dim = p_mean.shape[-1]
    batch_dim = p_mean.shape[0]

    # sum_k -0.5 * log (2 pi |p_cov|)
    term_one = -0.5 * torch.sum(torch.log(2 * np.pi * torch.linalg.det(p_cov)))

    p_precision = torch.linalg.inv(p_cov)  # recall, precision = cov^{-1}

    # sum_k -0.5 Tr[p_precision (q_cov + q_mean q_mean^T)]
    term_two = -0.5 * torch.sum(
        torch.einsum(
            'bii->b',
            torch.einsum(
                'bij, bjk->bik',
                p_precision,
                q_cov + torch.einsum('bi, bj->bij',
                                     q_mean,
                                     q_mean))))

    # sum_k -0.5 * -2 * mean_p Precision_p mean_q
    term_three = -0.5 * -2. * torch.einsum(
        'bi,bij,bj',
        p_mean,
        p_precision,
        q_mean,
    )

    # sum_k -0.5 * mean_p Precision_p mean_p
    term_four = -0.5 * -2. * torch.einsum(
        'bi,bij,bj',
        p_mean,
        p_precision,
        p_mean,
    )
    if check_einsums:
        term_two_check = -0.5 * torch.sum(torch.stack(
            [torch.trace(torch.matmul(p_precision[k],
                                      torch.add(q_cov[k],
                                                torch.outer(q_mean[k], q_mean[k].T))))
             for k in range(batch_dim)]))
        assert torch.isclose(term_two, term_two_check)

        term_three_check = torch.sum(torch.stack(
            [-.5 * -2. * torch.inner(p_mean[k],
                                     torch.matmul(p_precision[k],
                                                  q_mean[k]))
             for k in range(batch_dim)]))
        assert torch.isclose(term_three, term_three_check)

        term_four_check = torch.sum(torch.stack(
            [-.5 * -2. * torch.inner(p_mean[k],
                                     torch.matmul(p_precision[k],
                                                  p_mean[k]))
             for k in range(batch_dim)]))
        assert torch.isclose(term_four, term_four_check)
    total = term_one + term_two + term_three + term_four
    assert_torch_no_nan_no_inf_is_real(total)
    return total


def expected_log_gaussian_under_linear_gaussian(observation: torch.Tensor,
                                                q_A_mean: torch.Tensor,
                                                q_A_cov: torch.Tensor,
                                                q_Z_mean: torch.Tensor,
                                                sigma_obs_squared: float = 1e0,
                                                check_einsums: bool = False) -> torch.Tensor:
    """
    Compute E_{q(A, Z)}[log p(x|A, Z)] where p(x) = Gaussian(Z @ A, sigma_obs^2 I)
    and q(A, Z) = q(A) q(Z) are both Gaussian.

    All inputs are expected to have a batch dimension.
    """
    obs_dim = observation.shape[0]
    num_features = q_A_mean.shape[0]

    # -0.5 * log (2 * pi * |sigma_o^2 I|)
    term_one = -0.5 * torch.log(
        2 * np.pi * torch.linalg.det(sigma_obs_squared * torch.eye(obs_dim)))

    # o^T o
    term_two = torch.inner(observation, observation)

    # -2. * E[z_n] E[A_n]^T o
    term_three = -2. * torch.einsum(
        'b, bd, d->',
        q_Z_mean,
        q_A_mean,
        observation)

    # \sum_k b_{nk} Tr[Sigma_{nk} + \mu_{nk} \mu_{nk}^T]
    term_four = torch.einsum(
        'b,b->',
        q_Z_mean,
        torch.einsum('bii->b', q_A_cov + torch.einsum('bi, bj->bij',
                                                      q_A_mean,
                                                      q_A_mean)))

    # \sum_{k, k': k \neq k'} b_{nk} b_{nk'} \mu_{nk}^T \mu_{nk'}]
    # I don't know how to exclude the k=k' elements in the double sum, so we'll subtract
    # those using another einsum
    term_five_all_pairs = torch.einsum(
        'i,j,ik,jk->ij',
        q_Z_mean,
        q_Z_mean,
        q_A_mean,
        q_A_mean)
    term_five_self_pairs = torch.trace(term_five_all_pairs)
    term_five = torch.sum(term_five_all_pairs) - term_five_self_pairs

    if check_einsums:
        term_three_check = -2. * torch.sum(torch.multiply(q_Z_mean,
                                                          torch.matmul(q_A_mean,
                                                                       observation)))
        # TODO: debug why this assertion fails on the 15th step on a subset of datasets
        try:
            assert torch.isclose(term_three, term_three_check)
        except AssertionError:
            logging.error(str(term_three - term_three_check))
            raise AssertionError

        term_five_all_pairs_sum_check = torch.sum(torch.stack(
            [q_Z_mean[k] * q_Z_mean[kprime] * torch.inner(q_A_mean[k], q_A_mean[kprime])
             for k in range(num_features)
             for kprime in range(num_features)]))
        # TODO: debug why this assertion fails on the 21st step on a subset of datasets
        try:
            assert torch.isclose(torch.sum(term_five_all_pairs), term_five_all_pairs_sum_check)
        except AssertionError:
            logging.error(str(term_five_all_pairs - term_five_all_pairs_sum_check))
            raise AssertionError

    total = term_one - 0.5 * (term_two + term_three + term_four + term_five) / sigma_obs_squared
    assert_torch_no_nan_no_inf_is_real(total)
    return total


def entropy_bernoulli(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of p(x) = Bernoulli(prob).
    
    All inputs are expected to have a batch dimension.
    """
    one_minus_probs = 1. - probs

    term_one = torch.multiply(probs, torch.log(probs))

    term_two = torch.multiply(one_minus_probs, torch.log(one_minus_probs))

    # 0 * log 0 should be 0, but torch will numerically set a NaN.
    # Therefore, we mask these entries.
    term_one[probs == 0.] = 0.
    term_two[one_minus_probs == 0.] = 0.

    entropy = -torch.sum(torch.add(term_one, term_two))
    assert_torch_no_nan_no_inf_is_real(entropy)
    assert entropy >= 0.
    return entropy


def entropy_gaussian(mean: torch.Tensor,
                     cov: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of p(x) = Gaussian(mean, cov).

    All inputs are expected to have a batch dimension.
    """

    dim = mean.shape[-1]

    entropy = 0.5 * torch.sum(
        dim + dim * torch.log(torch.full(fill_value=2., size=()) * np.pi)
        + torch.log(torch.linalg.det(cov)))
    assert_torch_no_nan_no_inf_is_real(entropy)
    assert entropy >= 0.
    return entropy


def logits_to_probs(logits):
    probs = 1. / (1. + torch.exp(-logits))
    assert_torch_no_nan_no_inf_is_real(probs)
    assert torch.all(probs >= 0.)
    return probs


def probs_to_logits(probs):
    assert torch.all(probs >= 0.)
    logits = - torch.log(1. / probs - 1.)
    assert_torch_no_nan_no_inf_is_real(logits)
    return logits
