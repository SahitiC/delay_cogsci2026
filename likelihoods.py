"""
module to calculate likelihood of data under each of the models and maximise
log likelihood (minimise negative log likelihood) to find best fitting params
"""

import task_structure
import mdp_algms
import numpy as np
import helper
import constants


def softmax_policy(a, beta):
    c = a - np.max(a)
    p = np.exp(beta*c) / np.sum(np.exp(beta*c))
    return p


def log_likelihood(params, data):
    """Compute the log likelihood for a given model and data."""

    nllkhd = likelihood_basic_model_transformed(
        params, constants.STATES, constants.ACTIONS, constants.HORIZON,
        constants.REWARD_THR, constants.REWARD_EXTRA,
        constants.REWARD_SHIRK, constants.BETA, constants.THR,
        constants.STATES_NO, data)

    return -nllkhd


def calculate_likelihood_single(data, Q_values, beta, T, actions):
    """
    calculate negative likelihood of data under model given optimal Q_values,
    beta, transitions and actions available
    """
    nllkhd = 0

    if isinstance(data[0], (int, np.integer)):
        data = [data]
    else:
        data = data

    for traj in data:

        for t in range(len(traj)-1):

            partial = 0
            # enumerate over all posible actions for the observed state
            for i_a, action in enumerate(actions[traj[t]]):

                partial += (
                    softmax_policy(Q_values[traj[t]]
                                   [:, t], beta)[action]
                    * T[traj[t]][action][traj[t+1]])

            nllkhd = nllkhd - np.log(partial)

    return nllkhd


def calculate_likelihood(data, Q_values, beta, T, actions):
    """
    calculate negative likelihood of data under model given optimal Q_values,
    beta, transitions and actions available
    """
    nllkhd = 0

    if isinstance(data[0], (int, np.integer)):
        data = [data]
    else:
        data = data

    for trajectories in data:

        for traj in trajectories:

            for t in range(len(traj)-1):

                partial = 0
                # enumerate over all posible actions for the observed state
                for i_a, action in enumerate(actions[traj[t]]):

                    partial += (
                        softmax_policy(Q_values[traj[t]]
                                       [:, t], beta)[action]
                        * T[traj[t]][action][traj[t+1]])

                nllkhd = nllkhd - np.log(partial)

    return nllkhd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def likelihood_basic_model(x,
                           states, actions, horizon,
                           reward_thr, reward_extra, reward_shirk,
                           beta, thr, states_no, data):
    """
    implement likelihood calculation for basic model
    """

    discount_factor = x[0]
    efficacy = x[1]
    effort_work = x[2]

    # define task structure
    reward_func = task_structure.reward_no_immediate(
        states, actions, reward_shirk)

    effort_func = task_structure.effort(states, actions, effort_work)

    total_reward_func_last = task_structure.reward_final(
        states, reward_thr, reward_extra, thr, states_no)

    total_reward_func = []
    for state_current in range(len(states)):

        total_reward_func.append(reward_func[state_current]
                                 + effort_func[state_current])

    T = task_structure.T_binomial(states, actions, efficacy)

    # optimal policy
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        states, actions, horizon, discount_factor,
        total_reward_func, total_reward_func_last, T)

    nllkhd = calculate_likelihood(data, Q_values, beta, T, actions)

    return nllkhd


def likelihood_basic_model_transformed(
        x, states, actions, horizon, reward_thr, reward_extra, reward_shirk,
        beta, thr, states_no, data):
    """
    implement likelihood calculation for basic model
    """

    discount_factor_unbounded = x[0]
    efficacy_unbounded = x[1]
    effort_work_unbounded = x[2]

    [discount_factor, efficacy, effort_work] = helper.trans_to_bounded(
        [discount_factor_unbounded, efficacy_unbounded, effort_work_unbounded],
        [(0, 1), (0, 1), (None, 0)])

    # define task structure
    reward_func = task_structure.reward_no_immediate(
        states, actions, reward_shirk)

    effort_func = task_structure.effort(states, actions, effort_work)

    total_reward_func_last = task_structure.reward_final(
        states, reward_thr, reward_extra, thr, states_no)

    total_reward_func = []
    for state_current in range(len(states)):

        total_reward_func.append(reward_func[state_current]
                                 + effort_func[state_current])

    T = task_structure.T_binomial(states, actions, efficacy)

    # optimal policy
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        states, actions, horizon, discount_factor,
        total_reward_func, total_reward_func_last, T)

    nllkhd = calculate_likelihood_single(data, Q_values, beta, T, actions)

    return nllkhd
