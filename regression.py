# %% imports
from sklearn.cross_decomposition import CCA
import seaborn as sns
import gen_data
import constants
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import nquad
import matplotlib.pyplot as plt
import ast
from scipy.stats import pearsonr
from scipy.stats import chi2
import matplotlib as mpl
mpl.rcParams['font.size'] = 16
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['axes.linewidth'] = 1

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

        nll = 0.0
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


def drop_nans(*arrays):
    stacked = np.column_stack(arrays)
    mask = np.isnan(stacked).any(axis=1)
    return tuple(arr[~mask] for arr in arrays)


def get_mucw(row):
    units = np.array(ast.literal_eval(
        row['delta progress weeks']))*2
    units_cum = np.array(ast.literal_eval(
        row['cumulative progress weeks']))*2
    if np.max(units_cum) > 14:
        a = np.where(units_cum >= 14)[0][0]
        arr = units[:a+1]
        if units_cum[a] > 14:
            arr[-1] = 14 - units_cum[a-1]
        mucw = np.sum(arr * np.arange(1, len(arr)+1)) / 14
        return mucw
    else:
        arr = units
        mucw = np.sum(arr * np.arange(1, len(arr)+1)) / np.sum(arr)
        return mucw


def get_mucw_simulated(trajectory):
    if np.max(trajectory) > 14:
        a = np.where(trajectory >= 14)[0][0]
        arr = trajectory[:a+1]
        if arr[-1] > 14:
            arr[-1] = 14
        mucw = (14*(len(arr)+1) - np.sum(arr))/14
        return mucw
    else:
        arr = trajectory
        mucw = (np.max(arr)*(len(arr)+1) - np.sum(arr))/np.max(arr)
        return mucw


def get_completion_week(row):
    hours = np.array(ast.literal_eval(
        row['cumulative progress weeks']))
    if np.max(hours) >= 7:
        return np.where(hours >= 7)[0][0]
    else:
        return np.nan


# %%

if __name__ == "__main__":

    # %% data

    data_relevant = pd.read_csv('data_preprocessed.csv', index_col=False)

    data_full = pd.read_csv('zhang_ma_data.csv',
                            index_col=False)

    data_full_filter = data_full[data_full['SUB_INDEX_194'].isin(
        data_relevant['SUB_INDEX_194'])]
    data_full_filter = data_full_filter.reset_index(drop=True)

    result_fit_mle = np.load(
        "fit_individual_mle.npy", allow_pickle=True)

    result_fit_params = np.array([result_fit_mle[i]['par_b']
                                  for i in range(len(result_fit_mle))])

    result_diag_hess = np.array([result_fit_mle[i]['hess_diag']
                                for i in range(len(result_fit_mle))])

    # %% correlations model agnostic

    discount_factors_log_empirical = np.array(
        data_full_filter['DiscountRate_lnk'])
    discount_factors_empirical = np.exp(discount_factors_log_empirical)
    proc_mean = np.array(data_full_filter['AcadeProcFreq_mean'])
    mucw = np.array(data_relevant.apply(get_mucw, axis=1))

    # with N=160
    y, x = drop_nans(proc_mean, discount_factors_empirical)
    pearsonr(y, x)

    # with N=93 who worked to complete their requirement but no more
    filter = []
    for i in range(len(data_relevant)):
        traj = np.array(ast.literal_eval(data_relevant.iloc[i, 3]))*2
        # those who completed exactly 14
        if np.max(traj) == 14:
            filter.append(i)
        # those who did more but only to cross 7
        # only those who did 15, not more
        # i.e, 14 shouldn't be in the trajectory,
        # they shouldn't have done more units after crossing 7
        elif np.max(traj) > 14 and np.max(traj) < 17:
            if 14 not in traj and len(np.unique(traj[traj > 14])) < 2:
                filter.append(i)

    disc_emp = discount_factors_empirical[filter]
    proc = proc_mean[filter]
    mucw_ = mucw[filter]

    y, x = drop_nans(proc, disc_emp)
    pearsonr(y, x)

    # %% remove ppts with negative hess

    valid_indices = np.where(np.all(result_diag_hess > 0, axis=1))[0]

    data = data_full_filter.iloc[valid_indices].reset_index(drop=True)
    data_weeks = data_relevant.iloc[valid_indices].reset_index(drop=True)
    fit_params = result_fit_params[valid_indices]
    diag_hess = result_diag_hess[valid_indices]
    par_names = [r'$\gamma$', r'$\eta$', r'$r_{\text{effort}}$']

    # %% plot params
    for i in range(3):
        plt.figure(figsize=(4, 4), dpi=300)
        plt.hist(fit_params[:, i], color='grey')
        plt.xlabel(par_names[i])
        sns.despine()
        plt.savefig(
            f'plots/vectors/pars_{i}.svg',
            format='svg', dpi=300)
        plt.show()

    plt.figure(figsize=(4, 4), dpi=300)
    a = fit_params[:, 0]
    a = np.where(a == 1, 0.999, a)
    plt.hist(1/(1-a), color='grey')
    plt.xlabel(r'$\frac{1}{1-\gamma}$')
    sns.despine()
    plt.savefig(
        'plots/vectors/par_disc_transf.svg',
        format='svg', dpi=300)
    plt.show()

    count = 0
    for i in range(3):
        for j in range(i+1):
            if i != j:
                plt.figure(figsize=(4, 4), dpi=300)
                plt.scatter(fit_params[:, i],
                            fit_params[:, j], color='grey')
                plt.xlabel(par_names[i])
                plt.ylabel(par_names[j])
                count += 1
                sns.despine()
                plt.savefig(
                    f'plots/vectors/pars_corr_{count}.svg',
                    format='svg', dpi=300)
                plt.show()

    # %% variables

    # survey scores
    discount_factors_log_empirical = np.array(data['DiscountRate_lnk'])
    discount_factors_empirical = np.exp(discount_factors_log_empirical)
    proc_mean = np.array(data['AcadeProcFreq_mean'])
    impulsivity_score = np.array(data['ImpulsivityScore'])
    time_management = np.array(data['ReasonProc_TimeManagement'])
    task_aversiveness = np.array(data['ReasonProc_TaskAversiveness'])
    laziness = np.array(data['ReasonProc_Laziness'])
    self_control = np.array(data['SelfControlScore'])

    # fitted parameters
    discount_factors_fitted = fit_params[:, 0]
    efficacy_fitted = fit_params[:, 1]
    efforts_fitted = fit_params[:, 2]

    # delays
    mucw = np.array(data_weeks.apply(get_mucw, axis=1))
    completion_week = np.array(data_weeks.apply(get_completion_week, axis=1))

    # %% correlations

    # are impulsivity and self-control correlated with empirical discount
    # rates in the larger sample

    y, x = drop_nans(self_control, discount_factors_empirical)
    pearsonr(y, x)

    # %% plot mucw vs params

    np.random.seed(0)
    y, disc, efficacy, effort = drop_nans(
        mucw, discount_factors_fitted, efficacy_fitted,
        efforts_fitted)
    plt.figure(figsize=(4, 4), dpi=300)
    plt.scatter(disc, y, color='gray')
    plt.xlabel(r'$\gamma$')
    plt.ylabel('MUCW')
    sns.despine()
    plt.savefig(
        f'plots/vectors/mucw_disc.svg',
        format='svg', dpi=300)

    plt.figure(figsize=(4, 4), dpi=300)
    plt.scatter(efficacy, y, color='gray')
    plt.xlabel(r'$\eta$')
    plt.ylabel('MUCW')
    sns.despine()
    plt.savefig(
        f'plots/vectors/mucw_effic.svg',
        format='svg', dpi=300)
    plt.figure(figsize=(4, 4), dpi=300)
    plt.scatter(effort, y, color='gray')
    plt.xlabel(r'$r_{\text{effort}}$')
    plt.ylabel('MUCW')
    sns.despine()
    plt.savefig(
        f'plots/vectors/mucw_effort.svg',
        format='svg', dpi=300)

    a = disc
    a = np.where(disc == 1, 0.999, disc)
    plt.figure(figsize=(4, 4), dpi=300)
    plt.scatter(1/(1-a), y, color='gray')
    plt.xlabel(r'$\frac{1}{1-\gamma}$')
    plt.ylabel('MUCW')
    sns.despine()
    plt.savefig(
        f'plots/vectors/mucw_inv_disc.svg',
        format='svg', dpi=300)

    #  compare with simulated data for these parameters
    mucw_simulated = []
    for i in range(len(disc)):
        data = gen_data.gen_data_basic(
            constants.STATES, constants.ACTIONS, constants.HORIZON,
            constants.REWARD_THR, constants.REWARD_EXTRA,
            constants.REWARD_SHIRK, constants.BETA, disc[i], efficacy[i],
            effort[i], 5, constants.THR, constants.STATES_NO)
        temp = []
        for d in data:
            temp.append(get_mucw_simulated(d))
        mucw_i = np.nanmean(np.array(temp))
        mucw_simulated.append(mucw_i)
    plt.figure(figsize=(4, 4), dpi=300)
    plt.scatter(disc, mucw_simulated, color='gray')
    plt.xlabel(r'$\gamma$')
    plt.ylabel('MUCW')
    sns.despine()
    plt.savefig(
        f'plots/vectors/mucw_sim_disc.svg',
        format='svg', dpi=300)
    plt.figure(figsize=(4, 4), dpi=300)
    plt.scatter(1/(1-a), mucw_simulated, color='gray')
    plt.xlabel(r'$\frac{1}{1-\gamma}$')
    plt.ylabel('MUCW')
    sns.despine()
    plt.savefig(
        f'plots/vectors/mucw_sim_inv_disc.svg',
        format='svg', dpi=300)
    plt.figure(figsize=(4, 4), dpi=300)
    plt.scatter(efficacy, mucw_simulated, color='gray')
    plt.xlabel(r'$\eta$')
    plt.ylabel('MUCW')
    sns.despine()
    plt.savefig(
        f'plots/vectors/mucw_sim_effic.svg',
        format='svg', dpi=300)
    plt.figure(figsize=(4, 4), dpi=300)
    plt.scatter(effort, mucw_simulated, color='gray')
    plt.xlabel(r'$r_{\text{effort}}$')
    plt.ylabel('MUCW')
    sns.despine()
    plt.savefig(
        f'plots/vectors/mucw_sim_effort.svg',
        format='svg', dpi=300)

    # %% 3D plots

    trajectories = np.array(
        [ast.literal_eval(data_weeks['cumulative progress weeks'][i])
         for i in range(len(data_weeks))])*2
    units_completed = np.array([np.max(trajectories[i])
                                for i in range(len(trajectories))])

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d', elev=-155, azim=-45)
    p = ax.scatter(fit_params[:, 2], fit_params[:, 1],
                   fit_params[:, 0], c=units_completed, s=60, cmap='viridis')
    ax.set_title('Units completed', fontsize=14)
    ax.set_xlabel(r'$r_{\text{effort}}$', fontsize=14)
    ax.set_ylabel(r'$\eta$', fontsize=14)
    ax.set_zlabel(r'$\gamma$', fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='z', labelsize=14)
    ax.set_box_aspect(None, zoom=1.0)
    cbar = fig.colorbar(p)
    cbar.ax.tick_params(labelsize=14)
    plt.show()
    plt.savefig(
        f'plots/vectors/3D_1.svg',
        format='svg', dpi=300)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d', elev=-155, azim=-45)
    p = ax.scatter(fit_params[:, 2], fit_params[:, 1],
                   fit_params[:, 0], c=completion_week, s=60, cmap='viridis')
    ax.set_title('Completion week', fontsize=14)
    ax.set_xlabel(r'$r_{\text{effort}}$', fontsize=14)
    ax.set_ylabel(r'$\eta$', fontsize=14)
    ax.set_zlabel(r'$\gamma$', fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='z', labelsize=14)
    ax.set_box_aspect(None, zoom=1.0)
    cbar = fig.colorbar(p)
    cbar.ax.tick_params(labelsize=14)
    plt.show()
    plt.savefig(
        f'plots/vectors/3D_2.svg',
        format='svg', dpi=300)

    # %% regressions

    y, xhat, hess = drop_nans(
        self_control, discount_factors_fitted, diag_hess[:, 0])

    xhat_reshaped = xhat.reshape(-1, 1)

    # error regression with one predictor
    # bounds: for integration of latent parameter
    # opt_bounds: bounds for scipy optimise
    #             for free params (betas, intercept, sigma)
    result = fit_regression(y, xhat_reshaped,
                            (1/hess)**0.5,
                            bounds=[(0, 1)],
                            opt_bounds=[
                                (None, None), (None, None), (1e-3, None)],
                            initial_guess=[0.1, 0.1, 1])
    print(result)

    # null regression model with only intercept (and sigma_y ofc)
    result_null = fit_null_regression(y, xhat_reshaped,
                                      (1/hess)**0.5,
                                      bounds=[(0, 1)],
                                      opt_bounds=[(None, None), (1e-3, None)],
                                      initial_guess=[0.1, 1])
    print(result_null)

    # LRT
    lr_stat = 2 * (result_null.fun - result.fun)
    p_value = 1 - chi2.cdf(lr_stat, df=1)
    print(lr_stat, p_value)

    # %% mixed regressions (w error and exact terms)

    y, z, xhat, hess = drop_nans(
        self_control, discount_factors_empirical, discount_factors_fitted,
        diag_hess[:, 0])

    xhat_reshaped = xhat.reshape(-1, 1)
    z_reshaped = z.reshape(-1, 1)
    # regression with one latent predictor and one fixed
    result = fit_regression_mixed(y, z_reshaped, xhat_reshaped,
                                  (1/hess)**0.5,
                                  bounds=[(0, 1)],
                                  opt_bounds=[(None, None), (None, None),
                                              (None, None), (1e-3, None)],
                                  initial_guess=[0.1, 0.1, 0.1, 1])
    print(result)

    # null regression with coefficient for z set to zero
    result_null_z = fit_null_regression_mixed(
        y, z_reshaped, xhat_reshaped, (1/hess)**0.5,
        bounds=[(0, 1)],
        opt_bounds=[(None, None), (None, None), (1e-3, None)],
        initial_guess=[0.1, 0.1, 1], which_zero='z')
    print(result_null_z)
    # LRT
    lr_stat = 2 * (result_null_z.fun - result.fun)
    p_value = 1 - chi2.cdf(lr_stat, df=1)
    print(lr_stat, p_value)

    # null regression with coefficient for x set to zero
    result_null_x = fit_null_regression_mixed(
        y, z_reshaped, xhat_reshaped, (1/hess)**0.5,
        bounds=[(0, 1)],
        opt_bounds=[(None, None), (None, None), (1e-3, None)],
        initial_guess=[0.1, 0.1, 1], which_zero='x')
    print(result_null_x)
    # LRT
    lr_stat = 2 * (result_null_x.fun - result.fun)
    p_value = 1 - chi2.cdf(lr_stat, df=1)
    print(lr_stat, p_value)

    # %% cca

    df = pd.DataFrame({'pass': proc_mean,
                       'disc_emp': discount_factors_empirical,
                       'impulsivity': impulsivity_score,
                       'self_control': self_control,
                       'time_man': time_management,
                       'task_avers': task_aversiveness,
                       'disc': discount_factors_fitted,
                       'efficacy': efficacy_fitted,
                       'effort': efforts_fitted})

    df = df.dropna()
    df = (df-df.mean())/df.std()

    X = df.iloc[:, 0:6]
    Y = df.iloc[:, 6:9]

    cca = CCA(n_components=2)
    cca.fit(X, Y)
    X_c, Y_c = cca.transform(X, Y)
    score = cca.score(X, Y)

    plt.figure()
    plt.scatter(X_c[:, 0], Y_c[:, 0])
    print(pearsonr(X_c[:, 0], Y_c[:, 0]))
    plt.figure()
    plt.scatter(X_c[:, 1], Y_c[:, 1])
    print(pearsonr(X_c[:, 1], Y_c[:, 1]))

    print(cca.x_loadings_)
    print(cca.y_loadings_)
    print(cca.x_weights_)

    # variance explained
    print(np.mean(cca.x_loadings_**2, axis=0))
    print(np.mean(cca.y_loadings_**2, axis=0))


# %%
