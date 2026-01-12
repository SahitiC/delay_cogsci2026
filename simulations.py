# %%
import gen_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['font.size'] = 16
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['axes.linewidth'] = 1

# %%


def plot_trajectories(trajectories, color, lwidth_mean, lwidth_sample,
                      number_samples):
    mean = np.mean(trajectories, axis=0)

    plt.plot(mean, color=color, linewidth=lwidth_mean)
    for i in range(number_samples):
        plt.plot(trajectories[i], color=color,
                 linewidth=lwidth_sample, linestyle='dashed')
    plt.xticks([0, 7, 15])
    plt.yticks([0, 10, 22])
    plt.ylim(-1, 23)
    plt.xlabel('time in weeks')
    plt.ylabel('units of work completed')
    sns.despine()


# %%
np.random.seed(0)

cmap = plt.get_cmap('Blues')
colors = [0.4, 0.7, 0.9]
discounts = [0.4, 0.95, 0.99]
efficacy = 0.9
effort = -0.3
plt.figure(figsize=(4, 4))
for i_d, discount in enumerate(discounts):
    data = gen_data.simulate(
        np.array([[discount, efficacy, effort]]),
        n_trials=5, n_participants=1)
    plot_trajectories(np.squeeze(data),
                      cmap(colors[i_d]),
                      lwidth_mean=2, lwidth_sample=0.75,
                      number_samples=3)
    plt.title('Simulations')

# %%
cmap = plt.get_cmap('Greens')
colors = [0.4, 0.7, 0.9]
discount = 0.99
efficacys = [0.4, 0.7, 1]
effort = -0.3
plt.figure(figsize=(4, 4))
for i_ec, efficacy in enumerate(efficacys):
    data = gen_data.simulate(
        np.array([[discount, efficacy, effort]]),
        n_trials=5, n_participants=1)
    plot_trajectories(np.squeeze(data),
                      cmap(colors[i_ec]),
                      lwidth_mean=2, lwidth_sample=0.75,
                      number_samples=3)

# %%
cmap = plt.get_cmap('Oranges')
colors = [0.4, 0.7, 0.9]
discount = 0.99
efficacy = 0.9
efforts = [-3, -1.5, -0.3]
plt.figure(figsize=(4, 4))
for i_et, effort in enumerate(efforts):
    data = gen_data.simulate(
        np.array([[discount, efficacy, effort]]),
        n_trials=5, n_participants=1)
    plot_trajectories(np.squeeze(data),
                      cmap(colors[i_et]),
                      lwidth_mean=2, lwidth_sample=0.75,
                      number_samples=3)
# %%
