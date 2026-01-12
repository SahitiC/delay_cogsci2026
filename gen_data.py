"""
module to simulate data (trajectories) for each model given
input parameters and task structure
"""

import mdp_algms
import task_structure
import numpy as np
import helper
import constants


def softmax_policy(a, beta):
    c = a - np.max(a)
    p = np.exp(beta*c) / np.sum(np.exp(beta*c))
    return p


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def simulate(params, n_trials, n_participants):

    data = []
    for i in range(n_participants):
        [discount_factor, efficacy, effort_work] = params[i]
        datum = gen_data_basic(
            constants.STATES, constants.ACTIONS,  constants.HORIZON,
            constants.REWARD_THR, constants.REWARD_EXTRA,
            constants.REWARD_SHIRK, constants.BETA, discount_factor, efficacy,
            effort_work, n_trials, constants.THR, constants.STATES_NO)
        data.append(datum)
    return data


def gen_data_basic(states, actions, horizon, reward_thr, reward_extra,
                   reward_shirk, beta, discount_factor, efficacy, effort_work,
                   n_trials, thr, states_no):
    """
    function to generate a trajectory of state and action sequences given 
    parameters and reward, transition models of the basic model
    """

    # get reward function
    reward_func = task_structure.reward_no_immediate(
        states, actions, reward_shirk)

    effort_func = task_structure.effort(states, actions, effort_work)

    # reward delivered at the end of the semester
    total_reward_func_last = task_structure.reward_final(
        states, reward_thr, reward_extra, thr, states_no)

    total_reward_func = []
    for state_current in range(len(states)):

        total_reward_func.append(reward_func[state_current]
                                 + effort_func[state_current])

    # get tranistions
    T = task_structure.T_binomial(states, actions, efficacy)

    # get policy
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        states, actions, horizon, discount_factor,
        total_reward_func, total_reward_func_last, T)

    # generate data - forward runs
    initial_state = 0
    data = []
    for i_trials in range(n_trials):

        s, a = mdp_algms.forward_runs_prob(
            softmax_policy, Q_values, actions, initial_state, horizon, states,
            T, beta)
        data.append(s)

    return data


def gen_data_basic_transformed(
        states, actions, horizon, reward_thr, reward_extra, reward_shirk, beta,
        discount_factor_unbounded, efficacy_unbounded, effort_work_unbounded,
        n_trials, thr, states_no):
    """
    function to generate a trajectory of state and action sequences given 
    parameters and reward, transition models of the basic model
    """

    [discount_factor, efficacy, effort_work] = helper.trans_to_bounded(
        [discount_factor_unbounded, efficacy_unbounded, effort_work_unbounded],
        [(0, 1), (0, 1), (None, 0)])

    # get reward function
    reward_func = task_structure.reward_no_immediate(
        states, actions, reward_shirk)

    effort_func = task_structure.effort(states, actions, effort_work)

    # reward delivered at the end of the semester
    total_reward_func_last = task_structure.reward_final(
        states, reward_thr, reward_extra, thr, states_no)

    total_reward_func = []
    for state_current in range(len(states)):

        total_reward_func.append(reward_func[state_current]
                                 + effort_func[state_current])

    # get tranistions
    T = task_structure.T_binomial(states, actions, efficacy)

    # get policy
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        states, actions, horizon, discount_factor,
        total_reward_func, total_reward_func_last, T)

    # generate data - forward runs
    initial_state = 0
    data = []
    for i_trials in range(n_trials):

        s, a = mdp_algms.forward_runs_prob(
            softmax_policy, Q_values, actions, initial_state, horizon, states,
            T, beta)
        data.append(s)

    return data
