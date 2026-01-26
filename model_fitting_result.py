# %%
import seaborn as sns
import gen_data
import constants
import helper
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
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

# %% plot mucw vs params

discount_factors_fitted = fit_params[:, 0]
efficacy_fitted = fit_params[:, 1]
efforts_fitted = fit_params[:, 2]

mucw = np.array(data_weeks.apply(helper.get_mucw, axis=1))
completion_week = np.array(
    data_weeks.apply(helper.get_completion_week, axis=1))


np.random.seed(0)
y, disc, efficacy, effort = helper.drop_nans(
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
        temp.append(helper.get_mucw_simulated(d))
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

# %%
