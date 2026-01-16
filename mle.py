# %%
import numpy as np
from scipy.optimize import minimize
import likelihoods
import constants
import helper

# %%


def get_num_params(model_name):
    model_params = {'basic': 3}
    return model_params[model_name]


def get_param_ranges(model_name):
    param_ranges_dict = {
        'basic': [(0, 1), (0, 1), (None, 0)]}
    return param_ranges_dict[model_name]


def compute_log_likelihood(params, data, model_name):
    """Compute the log likelihood for a given model and data."""

    if model_name == 'basic':
        nllkhd = likelihoods.likelihood_basic_model(
            params, constants.STATES, constants.ACTIONS, constants.HORIZON,
            constants.REWARD_THR, constants.REWARD_EXTRA,
            constants.REWARD_SHIRK, constants.BETA, constants.THR,
            constants.STATES_NO, data)

    return nllkhd


def sample_initial_params(model_name, num_samples=1):
    """Sample initial parameters for MAP estimation."""

    if model_name == 'basic':
        discount_factor = np.random.uniform(0.0, 1)
        efficacy = np.random.uniform(0.0, 1)
        effort_work = -1 * np.random.exponential(0.5)
        pars = [discount_factor, efficacy, effort_work]

    return pars


def sample_params(model_name, num_samples=1):
    """Sample parameters to generate data."""

    if model_name == 'basic':
        discount_factor = np.random.uniform(0.2, 1)
        efficacy = np.random.uniform(0.35, 1)
        effort_work = -1 * np.random.exponential(0.5)
        pars = [discount_factor, efficacy, effort_work]

    return pars


def MLE(data_participant, model_name, iters=5, initial_guess=None):
    """
    Maximim Likelihood estimate (MLE) for a single participant.

    Parameters
    ----------
    data_participant : list
        List of data arrays for a single participant.
    model_name : str
        Name of the model to fit. 'basic' is currently supported.
    iters : int
        Number of random initialisations for optimization.
    initial_guess : list or None
        Initial guess for parameters. If None, random initialisation is used.

    Returns
    -------
    fit_participant : dict
        Dictionary containing fitted parameters, Hessian diagonal,
        negative log likelihood, and success flag.
    """

    param_ranges = get_param_ranges(model_name)

    # negative log posterior
    def neg_log_lik(pars):

        neg_log_lik = compute_log_likelihood(
            pars, data_participant, model_name)

        return neg_log_lik

    # optimization
    nllkhd = np.inf

    # with initial guess
    if initial_guess is not None:
        pars = initial_guess
        res = minimize(neg_log_lik, pars, bounds=param_ranges)
        if res.fun < nllkhd:
            nllkhd = res.fun
            final_res = res
        else:
            print(res.fun)

    # iterate with random initialisations
    for iter in range(iters):
        pars = sample_initial_params(model_name)
        res = minimize(neg_log_lik, pars, bounds=param_ranges)
        if res.fun < nllkhd:
            nllkhd = res.fun
            final_res = res

    diag_hess = helper.Hess_diag(neg_log_lik, final_res.x)

    fit_participant = {'par_b': final_res.x,
                       'hess_diag': diag_hess,
                       'neg_log_lik': final_res.fun,
                       'success': final_res.success}

    return fit_participant
