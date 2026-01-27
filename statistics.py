# %%
from sklearn.cross_decomposition import CCA
import helper
import regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
from scipy.stats import pearsonr
from scipy.stats import chi2
import matplotlib as mpl
mpl.rcParams['font.size'] = 16
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['axes.linewidth'] = 1

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
mucw = np.array(data_relevant.apply(helper.get_mucw, axis=1))

# with N=160, do all three combinations
y, x = helper.drop_nans(proc_mean, discount_factors_empirical)
print(pearsonr(y, x))

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

y, x = helper.drop_nans(proc, disc_emp)
print(pearsonr(y, x))

# %% remove ppts with negative hess

valid_indices = np.where(np.all(result_diag_hess > 0, axis=1))[0]

data = data_full_filter.iloc[valid_indices].reset_index(drop=True)
data_weeks = data_relevant.iloc[valid_indices].reset_index(drop=True)
fit_params = result_fit_params[valid_indices]
diag_hess = result_diag_hess[valid_indices]
par_names = [r'$\gamma$', r'$\eta$', r'$r_{\text{effort}}$']

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
mucw = np.array(data_weeks.apply(helper.get_mucw, axis=1))
completion_week = np.array(data_weeks.apply(
    helper.get_completion_week, axis=1))

# %% regressions of mucw, proc_mean with fitted params

np.random.seed(0)

y, x1, x2, x3, hess = helper.drop_nans(proc_mean, discount_factors_fitted,
                                       efficacy_fitted, efforts_fitted,
                                       diag_hess)

xhat = np.column_stack((x1, x2, x3))

bounds = [(0, 1), (0, 1), (-4, 0)]
x_samples = np.column_stack([np.random.uniform(a, b, size=5000)
                             for (a, b) in bounds])

result = regression.fit_regression_mc(
    y, xhat, (1/hess)**0.5,
    bounds=[(0, 1), (0, 1), (-4, 0)],
    x_samples=x_samples,
    opt_bounds=[(None, None), (None, None), (None, None),
                (None, None), (1e-3, None)],
    initial_guess=[1, 1, 1, 0.1, 1])

# null models
results_null = []
for i in range(3):
    results_null.append(regression.fit_restricted_regression_mc(
        y, xhat, (1/hess)**0.5,
        bounds=[(0, 1), (0, 1), (-4, 0)],
        x_samples=x_samples,
        opt_bounds=[(None, None), (None, None),
                    (None, None), (1e-3, None)],
        initial_guess=[1, 1, 0.1, 1],
        restricted_indices=[i]))

result_disc_only = regression.fit_restricted_regression_mc(
    y, xhat, (1/hess)**0.5,
    bounds=[(0, 1), (0, 1), (-4, 0)],
    x_samples=x_samples,
    opt_bounds=[(None, None), (None, None), (1e-3, None)],
    initial_guess=[1, 0.1, 1],
    restricted_indices=[1, 2])

for i in range(3):
    lr_stat = 2 * (results_null[i].fun - result.fun)
    p_value = 1 - chi2.cdf(lr_stat, df=1)
    print(lr_stat, p_value)

lr_stat = 2 * (result_disc_only.fun - result.fun)
p_value = 1 - chi2.cdf(lr_stat, df=2)
print(lr_stat, p_value)

# %% regressions

y, xhat, hess = helper.drop_nans(
    discount_factors_empirical, discount_factors_fitted, diag_hess[:, 0])

xhat_reshaped = xhat.reshape(-1, 1)

# error regression with one predictor
# bounds: for integration of latent parameter
# opt_bounds: bounds for scipy optimise
#             for free params (betas, intercept, sigma)
result = regression.fit_regression(y, xhat_reshaped,
                                   (1/hess)**0.5,
                                   bounds=[(0, 1)],
                                   opt_bounds=[
                                       (None, None), (None, None),
                                       (1e-3, None)],
                                   initial_guess=[0.1, 0.1, 1])
print(result)

# null regression model with only intercept (and sigma_y ofc)
result_null = regression.fit_null_regression(y, xhat_reshaped,
                                             (1/hess)**0.5,
                                             bounds=[(0, 1)],
                                             opt_bounds=[
                                                 (None, None), (1e-3, None)],
                                             initial_guess=[0.1, 1])
print(result_null)

# LRT
lr_stat = 2 * (result_null.fun - result.fun)
p_value = 1 - chi2.cdf(lr_stat, df=1)
print(lr_stat, p_value)

# %% mixed regressions (w error and exact terms)

y, z, xhat, hess = helper.drop_nans(
    impulsivity_score, discount_factors_empirical, discount_factors_fitted,
    diag_hess[:, 0])

xhat_reshaped = xhat.reshape(-1, 1)
z_reshaped = z.reshape(-1, 1)
# regression with one latent predictor and one fixed
result = regression.fit_regression_mixed(y, z_reshaped, xhat_reshaped,
                                         (1/hess)**0.5,
                                         bounds=[(0, 1)],
                                         opt_bounds=[(None, None), (None, None),
                                                     (None, None), (1e-3, None)],
                                         initial_guess=[0.1, 0.1, 0.1, 1])
print(result)

# null regression with coefficient for z set to zero
result_null_z = regression.fit_null_regression_mixed(
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
result_null_x = regression.fit_null_regression_mixed(
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

# %%
