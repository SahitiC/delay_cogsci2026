"""
define quantities that are held constant across scripts
"""

import numpy as np

# define some standard params:
# states of markov chain
STATES_NO = 22+1  # one extra state for completing nothing
STATES = np.arange(STATES_NO)

# actions = no. of units to complete in each state
# allow as many units as possible based on state
ACTIONS = [np.arange(STATES_NO-i) for i in range(STATES_NO)]

HORIZON = 16  # no. of weeks for task
DISCOUNT_FACTOR = 0.9  # discounting factor
EFFICACY = 0.8  # self-efficacy (probability of progress for each unit)

# utilities :
REWARD_THR = 4.0  # reward per unit at threshold (14 units)
REWARD_SHIRK = 0.1
EFFORT_WORK = -0.3
REWARD_EXTRA = REWARD_THR/8  # extra reward for completing all units
BETA = 5  # softmax beta
THR = 14
N_TRIALS = 20  # no. of trajectories per dataset for recovery
# N = 1000  # no of params sets to recover
