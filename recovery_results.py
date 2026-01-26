# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
fit_params = np.load('fit_params_mle.npy', allow_pickle=True)
recovered_fits = np.load("recovery_fits_mle.npy",
                         allow_pickle=True)

n_params = 3
if n_params == 3:
    lim = [(-0.05, 1.05), (-0.05, 1.05), (-5, 0.05)]
elif n_params == 2:
    lim = [(-0.05, 1.05), (-5, 0.05)]

tolerance = [0.45, 0.45, 1.8]

recovered_fit_params = np.stack([recovered_fits[i]['par_b']
                                 for i in range(len(fit_params))])

mask = np.where(fit_params[:, 0] != 0)
idxs = []  # to exclude
for i in range(n_params):
    colors = []
    for j in range(len(fit_params)):
        if np.abs(fit_params[j, i]-recovered_fit_params[j, i]) < tolerance[i]:
            colors.append('tab:blue')
        else:
            colors.append('tab:red')
            if not (j in idxs):
                idxs.append(j)

    colors = np.array(colors)
    print(sum(np.where(colors == 'tab:red', 1, 0)))
    plt.figure(figsize=(4, 4))
    plt.scatter(fit_params[mask, i],
                recovered_fit_params[mask, i],
                c=colors[mask])
    x = np.array([a for a in fit_params[mask, i]])
    y = np.array([a for a in recovered_fit_params[mask, i]])
    corr = np.corrcoef(x, y)
    plt.title(f'corr = {np.round(corr[0, 1], 3)}')
    plt.plot(
        np.linspace(lim[i][0], lim[i][1], 10),
        np.linspace(lim[i][0], lim[i][1], 10),
        linewidth=1, color='black')  # x=y line
    plt.xlim(lim[i])
    plt.ylim(lim[i])

    plt.xlabel('Input Parameter')
    plt.ylabel('Recovered Parameter')

# %%
