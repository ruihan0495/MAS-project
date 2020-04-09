import torch
import torch.nn as nn
import minority_game 
#import es
import utils
import gnn
import numpy as np
from collections import deque

"A script training the graph neural network uses evolutionary algorithm"
EPISODE_LENGTH = 15
NPOPULATION = 10
env = minority_game.MinorGameEnv(2)
model = gnn.TransitionGNN(input_dim = 2, hidden_dim = 6, action_dim = 1, num_agents = 5, output_dim = 1)
k = model.num_agents//2

def train(model, env):
    #num_parameters = utils.count_parameters(model)
    states_so_far = deque()
    aug_states_so_far = deque()
    '''
    solver = es.CMAES(num_parameters,
              popsize=NPOPULATION,
              weight_decay=0.0,
              sigma_init = 0.5
            )
    '''        
    state_i = env.get_observation()        
    for i in range(EPISODE_LENGTH):
        states_so_far.append(state_i)
        if i < model.num_time_steps:
            action_n = utils.random_action(model.num_agents)
            rewards, state_i = env.step(action_n)
            print(len(states_so_far))   
        # concat frames in past k time steps
        else:
            j = 1
            aug_state = states_so_far[i]
            while j < model.num_time_steps:
                aug_state = np.hstack((aug_state, states_so_far[i-j]))
                j+=1
            aug_states_so_far.append(aug_state)
            # feed into network
            input = aug_state[np.newaxis,:,:]           
            out = model(input)
            print(out)
            # on-policy update to be revised
            #action_n = utils.to_zero_one(out)
            #rewards, state_i = env.step(action_n)
            #n_a = np.count_nonzero(action_n)

            #fitness_score = utils.calc_fitness_score(n_a, k)
            #utils.test_solver(solver, )

if __name__ == "__main__":
    train(model, env)