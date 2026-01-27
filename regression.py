# %% imports
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import nquad
from scipy.special import logsumexp

# %% functions


def integrand(*args):
    """
    Multivariate normal integrand for regression with measurement error.
    Calculates integrand for each observation (y_i, x_hat_i) and some values of
    the latent predictors (x1, x2, x3).

    Parameters
    ----------
    args = (x1, x2, ..., xp, y_i, xhat_i, beta, intercept, sigma, sigma_x_i)
    x1, x2, ..., xp : float
        Variables of integration. Latent predictors in the regression model.
    y_i : float
        Observed dependent variable.
    xhat_i : array-like, shape (p,)           
        Observed estimates of the predictors with measurement error.
    beta : array-like, shape (p,)
        Regression coefficients.
    intercept : float
        Regression intercept.
    sigma : float
        Standard deviation of the measurement error in y.
    sigma_x_i : array-like, shape (p,)
        Standard deviations of the measurement errors in predictors. 
        Given by inverse hessian.
    """
    *xs, y_i, xhat_i, sigma_x_i, beta, intercept, sigma = args
    x = np.array(xs)
    # uniform = 1 / np.prod([b - a for a, b in bounds])
    integrand = (
                (1/(np.prod(sigma_x_i)*sigma)) *
        np.exp(-0.5 * np.sum(((x - xhat_i)/sigma_x_i)**2) +
                    -0.5 * ((y_i - (np.dot(beta, x) + intercept))/sigma)**2))
    return integrand


def integrand_mixed(*args):
    """
    z_i: exact predictors with no measurement error
    beta_x: coefficients for latent variables x
    beta_z: coefficients for latent variables z 
    rest: same as before
    """
    *xs, y_i, z_i, xhat_i, sigma_x_i, beta_z, beta_x, intercept, sigma = args
    x = np.array(xs)
    # uniform = 1 / np.prod([b - a for a, b in bounds])
    integrand = (
                (1/(np.prod(sigma_x_i)*sigma)) *
        np.exp(-0.5 * np.sum(((x - xhat_i)/sigma_x_i)**2) +
                    -0.5 * ((y_i - (np.dot(beta_z, z_i) +
                                    np.dot(beta_x, x) + intercept))/sigma)**2))
    return integrand


def likelihood_i(pars, y_i, xhat_i, sigma_x_i, bounds):
    """
    Likelihood for a single observation (y_i, x_hat_i) given parameters.

    Parameters
    ----------
    pars : array-like, shape (p+2,)
        Model parameters: [beta_1, beta_2, ..., beta_p, intercept, sigma]
    y_i : float
        Observed dependent variable.
    xhat_i : array-like, shape (p,)
        Observed estimates of the predictors with measurement error.
    sigma_x_i : array-like, shape (p,)
        Standard deviations of the measurement errors in predictors.
    bounds : list of tuples
        Integration bounds for each predictor.
    """
    p = len(xhat_i)  # no. of predictors
    beta = pars[0:p]
    intercept = pars[p]
    sigma = pars[p+1]

    integral, error = nquad(
        integrand, bounds,
        args=(y_i, xhat_i, sigma_x_i, beta, intercept, sigma))

    return integral


def likelihood_i_mixed(pars, y_i, z_i, xhat_i, sigma_x_i, bounds):

    p_z = len(z_i)  # no. of exact predictors
    p_x = len(xhat_i)  # no. of latent predictors
    beta_z = pars[0:p_z]
    beta_x = pars[p_z:p_z+p_x]
    intercept = pars[p_z+p_x]
    sigma = pars[p_z+p_x+1]

    integral, error = nquad(
        integrand_mixed, bounds,
        args=(y_i, z_i, xhat_i, sigma_x_i, beta_z, beta_x, intercept, sigma))

    return integral


def negative_log_likelihood(pars, y, xhat, sigma_x, bounds):
    """
    Negative log likelihood for the entire dataset.

    Parameters
    ----------
    pars : array-like, shape (p+2,)
        Model parameters: [beta_1, beta_2, ..., beta_p, intercept, sigma]
    y : array-like, shape (n_observations,)
        Observed dependent variable.
    xhat : array-like, shape (n_observations, p)
        Observed estimates of the predictors with measurement error.
    sigma_x : array-like, shape (n_observations, p)
        Standard deviations of the measurement errors in predictors.
    bounds : list of tuples
        Integration bounds for each predictor.
    """

    nll = 0
    for i in range(len(y)):
        ll_i = likelihood_i(pars, y[i], xhat[i], sigma_x[i], bounds)
        nll -= np.log(ll_i + 1e-10)  # add small constant to avoid log(0)
    return nll


def negative_log_likelihood_mixed(pars, y, z, xhat, sigma_x, bounds):

    nll = 0
    for i in range(len(y)):
        ll_i = likelihood_i_mixed(
            pars, y[i], z[i], xhat[i], sigma_x[i], bounds)
        nll -= np.log(ll_i + 1e-10)  # add small constant to avoid log(0)
    return nll


def fit_regression(y, xhat, sigma_x, bounds, opt_bounds, initial_guess):
    """
    Fit regression model using MLE.

    Parameters
    ----------
    y : array-like, shape (n_observations,)
        Observed dependent variable.
    xhat : array-like, shape (n_observations, p)
        Observed estimates of the predictors with measurement error.
    sigma_x : array-like, shape (n_observations, p)
        Standard deviations of the measurement errors in predictors.
    initial_guess : array-like, shape (p+2,), optional
        Initial guess for the parameters: [beta_1, beta_2, ..., beta_p,
        intercept, sigma]
    bounds : list of tuples
        Integration bounds for each predictor.
    opt_bounds : list of tuples
        Bounds for the optimization parameters.

    Returns
    -------
    result : OptimizeResult
        The optimization result represented as a `OptimizeResult` object.
    """
    result = minimize(negative_log_likelihood, initial_guess,
                      args=(y, xhat, sigma_x, bounds),
                      bounds=opt_bounds)
    return result


def fit_regression_mixed(y, z, xhat, sigma_x, bounds, opt_bounds,
                         initial_guess):

    result = minimize(negative_log_likelihood_mixed, initial_guess,
                      args=(y, z, xhat, sigma_x, bounds),
                      bounds=opt_bounds)
    return result


def fit_null_regression(y, xhat, sigma_x, bounds, opt_bounds, initial_guess):
    """
    Fit null regression model (intercept only) using MLE.

    Parameters
    ----------
    y : array-like, shape (n_observations,)
        Observed dependent variable.
    opt_bounds : list of tuples
        Bounds for the optimization parameters.
    initial_guess : array-like, shape (2,), optional
        Initial guess for the parameters: [intercept, sigma] 
    Returns
    -------
    result : OptimizeResult
        The optimization result represented as a `OptimizeResult` object.
    """

    def negative_log_likelihood_null(pars, y, xhat, sigma_x, bounds):
        # pars = [intercept, sigma]
        p = xhat.shape[1]
        beta = np.zeros(p)
        intercept, sigma = pars

        nll = 0
        for i in range(len(y)):
            ll_i = likelihood_i(
                np.r_[beta, intercept, sigma],
                y[i], xhat[i], sigma_x[i], bounds)
            nll -= np.log(ll_i + 1e-10)
        return nll

    result = minimize(negative_log_likelihood_null, initial_guess,
                      args=(y, xhat, sigma_x, bounds),
                      bounds=opt_bounds)
    return result


def fit_null_regression_mixed(y, z, xhat, sigma_x, bounds, opt_bounds,
                              initial_guess, which_zero='x'):

    def negative_log_likelihood_null_mixed(pars, y, z, xhat, sigma_x, bounds,
                                           which_zero):

        p_z = z.shape[1]
        p_x = xhat.shape[1]

        if which_zero == 'x':
            beta_x = np.zeros(p_x)
            beta_z = pars[:p_z]
            intercept = pars[p_z]
            sigma = pars[p_z + 1]

        elif which_zero == 'z':
            beta_z = np.zeros(p_z)
            beta_x = pars[:p_x]
            intercept = pars[p_x]
            sigma = pars[p_x + 1]
        nll = 0.0
        for i in range(len(y)):
            ll_i = likelihood_i_mixed(
                np.r_[beta_z, beta_x, intercept, sigma],
                y[i], z[i], xhat[i], sigma_x[i], bounds)
            nll -= np.log(ll_i + 1e-10)
        return nll

    result = minimize(negative_log_likelihood_null_mixed, initial_guess,
                      args=(y, z, xhat, sigma_x, bounds, which_zero),
                      bounds=opt_bounds)
    return result


# implement error regressions with monte carlo integration
def log_gaussian(y, mean, sigma):
    return (
        - np.log(sigma)
        - 0.5 * ((y - mean) / sigma)**2)


def log_gaussian_vec(x, xhat, sigma_x):
    return (
        - np.log(np.prod(sigma_x))
        - 0.5 * np.sum(((xhat - x)/sigma_x)**2, axis=1))


def log_likelihood_i_mc(pars, y_i, xhat_i, sigma_x_i, bounds,
                        x_samples):
    """
    Likelihood for a single observation (y_i, x_hat_i) given parameters using
    Monte Carlo integration.

    Parameters
    ----------
    pars : array-like, shape (p+2,)
        Model parameters: [beta_1, beta_2, ..., beta_p, intercept, sigma]
    y_i : float
        Observed dependent variable.
    xhat_i : array-like, shape (p,)
        Observed estimates of the predictors with measurement error.
    sigma_x_i : array-like, shape (p,)
        Standard deviations of the measurement errors in predictors.
    bounds : list of tuples
        Integration bounds for each predictor.
    sample_size : int, optional
        Number of Monte Carlo samples to draw. Default is 1000.

    Returns
    -------
    float
        Log-likelihood for individual i.
    """

    p = len(xhat_i)  # no. of predictors
    beta = pars[0:p]
    intercept = pars[p]
    sigma = pars[p+1]

    log_pxhat = log_gaussian_vec(x_samples, xhat_i, sigma_x_i)
    mean = x_samples @ beta + intercept
    log_py = log_gaussian(y_i, mean, sigma)

    return logsumexp(log_pxhat + log_py) - np.log(len(x_samples))


def negative_log_likelihood_mc(pars, y, xhat, sigma_x, bounds,
                               x_samples):
    """
    Negative log likelihood for the entire dataset. Avoid for more than one
    predictor x_i due to compuational cost of integration.

    Parameters
    ----------
    pars : array-like, shape (p+2,)
        Model parameters: [beta_1, beta_2, ..., beta_p, intercept, sigma]
    y : array-like, shape (n_observations,)
        Observed dependent variable.
    xhat : array-like, shape (n_observations, p)
        Observed estimates of the predictors with measurement error.
    sigma_x : array-like, shape (n_observations, p)
        Standard deviations of the measurement errors in predictors.
    bounds : list of tuples
        Integration bounds for each predictor.
    sample_size_mc : int, optional
        Number of Monte Carlo samples to draw. Default is 1000.

    Returns
    -------
    float
        Negative log-likelihood for the dataset.
    """

    nll = 0
    for i in range(len(y)):
        nll -= log_likelihood_i_mc(
            pars, y[i], xhat[i], sigma_x[i], bounds,
            x_samples)
    return nll


def fit_regression_mc(y, xhat, sigma_x, bounds, x_samples,
                      opt_bounds, initial_guess):

    result = minimize(negative_log_likelihood_mc, initial_guess,
                      args=(y, xhat, sigma_x, bounds, x_samples),
                      bounds=opt_bounds)
    return result


def fit_restricted_regression_mc(y, xhat, sigma_x, bounds, x_samples,
                                 opt_bounds, initial_guess, restricted_indices):

    def negative_log_likelihood_null(pars, y, xhat, sigma_x, bounds,
                                     restricted_indices):

        p = xhat.shape[1]
        beta = np.zeros(p)
        free_idx = ~np.isin(np.arange(p), restricted_indices)
        beta[free_idx] = pars[0:free_idx.sum()]
        intercept = pars[free_idx.sum()]
        sigma = pars[free_idx.sum()+1]

        pars_restricted = np.r_[beta, intercept, sigma]
        nll = negative_log_likelihood_mc(
            pars_restricted, y, xhat, sigma_x, bounds, x_samples)

        return nll

    result = minimize(negative_log_likelihood_null, initial_guess,
                      args=(y, xhat, sigma_x, bounds, restricted_indices),
                      bounds=opt_bounds)
    return result
