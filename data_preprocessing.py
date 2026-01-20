# %%

import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['font.size'] = 16
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['axes.linewidth'] = 1

# %% functions


def process_delta_progress(row, semester_length_weeks):
    """
    aggregate delta progress over days to weeks
    """
    temp = ast.literal_eval(row['delta progress'])
    temp_week = [sum(temp[i_week*7: (i_week+1)*7])
                 for i_week in range(semester_length_weeks)]

    assert sum(temp_week) == row['Total credits']
    return temp_week


def cumulative_progress_weeks(row):

    return np.cumsum(row['delta progress weeks']).tolist()


def get_timeseries_to_cluster(row):

    return row['cumulative progress weeks']


# %% drop unwanted rows

data = pd.read_csv('zhang_ma_data.csv')

# drop the ones that discontinued (subj. 1, 95, 111)
# they report to have discountinued in verbal response in 'way_allocate' column
# they do 1 hour in the very beginning and then nothing after
# sbj 24, 55, 126 also dont finish 7 hours but not because they drop out
data_relevant = data.drop([1, 95, 111])
data_relevant = data_relevant.reset_index(drop=True)

# drop NaN entries
data_relevant = data_relevant.dropna(subset=['delta progress'])
data_relevant = data_relevant.reset_index(drop=True)

# drop ones who complete more than 11 hours
# as extra credit ends at 11 hours
# we do not consider extra rewards for > 11 hours in our models as well
mask = np.where(data_relevant['Total credits'] <= 11)[0]
data_relevant = data_relevant.loc[mask]
data_relevant = data_relevant.reset_index(drop=True)

semester_length = len(ast.literal_eval(data_relevant['delta progress'][0]))

# %% transform trajectories

# delta progress week wise
semester_length_weeks = round(semester_length/7)
data_relevant['delta progress weeks'] = data_relevant.apply(
    lambda row: process_delta_progress(row, semester_length_weeks), axis=1)

# cumulative progress week wise
data_relevant['cumulative progress weeks'] = data_relevant.apply(
    cumulative_progress_weeks, axis=1)

# choose columns to save
data_subset = data_relevant[['SUB_INDEX_194', 'Total credits',
                             'delta progress', 'cumulative progress',
                             'delta progress weeks',
                             'cumulative progress weeks']]

data_subset.to_csv('data_preprocessed.csv', index=False)

# %%
plt.figure(figsize=(4, 4), dpi=300)
for i in [18, 65, 93, 139]:
    units = np.array(data_subset['cumulative progress weeks'][i])
    plt.plot(units*2, color='black')
plt.xlabel('time in weeks')
plt.ylabel('units of work completed')
plt.title('Data')
plt.xticks([0, 7, 15])
plt.yticks([0, 10, 22])
sns.despine()
plt.savefig(
    f'plots/vectors/data.svg',
    format='svg', dpi=300)
plt.show()

# %%
