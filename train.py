import torch
import torch.nn as nn
import minority_game as env
import es
import utils
import gnn
import numpy as np
from collections import deque

"A script training the graph neural network uses evolutionary algorithm"
EPISODE_LENGTH = 15
NPOPULATION = 10

def train(model, env):
    num_parameters = utils.count_parameters(model)
    states_so_far = deque(maxlen=EPISODE_LENGTH)
    aug_states_so_far = deque()
    solver = es.CMAES(num_parameters,
              popsize=NPOPULATION,
              weight_decay=0.0,
              sigma_init = 0.5
            )
    for i in range(EPISODE_LENGTH):
        state_i = env.get_observation()
        states_so_far.append(state_i)
        if i < model.num_time_steps:
            action_n = utils.random_action(model.num_agents)
            rewards, next_observations = env.step(action_n)
            states_so_far.append(next_observations)
        # concat frames in past k time steps
        else:
            j = 0
            aug_state = []
            while j < model.num_time_steps:
                aug_state = np.vstack(aug_state, states_so_far[i-j])
                j+=1
            aug_state = aug_state.view(-1,1)
            aug_states_so_far.append(aug_state)
            # feed into network
            out = model(aug_state)
            # on-policy update
            action_n = utils.to_zero_one(out)
            rewards, next_observations = env.step(action_n)
            states_so_far.append(next_observations)
            n_a = np.count_nonzero(action_n)

            fitness_score = utils.calc_fitness_score(n_a, model.k)
            utils.test_solver(solver, )
