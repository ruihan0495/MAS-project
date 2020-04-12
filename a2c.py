import torch
import torch.nn as nn
import numpy as np
import minority_game
import utils
import argparse
from collections import deque

'''
implement the a2c baseline here
'''

class A2C(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, num_time_steps):
        super(A2C, self).__init__()
        state_dim = state_dim*num_time_steps
        self.num_agents = 5
        self.num_time_steps = num_time_steps
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, input):
        policy_dist = self.actor(input)
        state_value = self.critic(input)
        return policy_dist, state_value 

    def select_action(self, actions, policy_dist, epsilon=0.05):
        # apply epsilon-greedy action selection
        is_random = np.random.rand()
        if is_random > epsilon:
            action = np.argmax(policy_dist.detach().numpy())    
        else:
            action = actions.sample()
        return action


def single_train(rollout_length, actor_lr, critic_lr, entropy_reg, gamma=0.2):
    env = minority_game.MinorGameEnv(2)
    model = A2C(2,6,2,5) 
    state_i = env.get_observation()   
    loss_fn = nn.MSELoss()
    states_so_far = deque()
    aug_states_so_far = deque()
    actor_optimizer = torch.optim.Adam(model.parameters(),lr=actor_lr)
    critic_optimizer = torch.optim.Adam(model.parameters(),lr=critic_lr)
    for i in range(rollout_length):
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
            input = torch.tensor(aug_state[np.newaxis,:,:])           
            dist, value = model(input)
            # sample next action
            next_action = model.select_action(env.action_space, dist)
            rewards, state_i = env.step([next_action]*5)
            # state_i has shape[num_agents, state_dim]
            hor_state_i = np.reshape(state_i,(1,-1))
            temp_aug_state = np.vstack((aug_state, hor_state_i))
            temp_aug_state = temp_aug_state[1:,:]
            input = torch.tensor(temp_aug_state)
            _, next_value = model(input)
            # train network
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            #np_dist = dist.detach().numpy()
            log_probs = torch.sum(torch.log(dist))
            advantange = torch.tensor(rewards) +  gamma*next_value - value
            entropy = torch.sum(-torch.mean(dist)*torch.log(dist))
            actor_loss =torch.mean(log_probs*advantange) + entropy_reg*entropy
            critic_loss = loss_fn(torch.tensor(rewards)+next_value, value)
            actor_loss.backward(retain_graph=True)
            critic_loss.backward()
            actor_optimizer.step()
            critic_optimizer.step()


parser = argparse.ArgumentParser()

## General parameters
parser.add_argument("--actor_lr", type=float, default=0.001,
                    help="learning rate for actor")
parser.add_argument("--critic_lr", type=float, default=0.001,
                    help="learning rate for critic")
parser.add_argument("--rollout_len", type=int, default=15,
                    help="rollout length of each episode")
parser.add_argument("--entropy_reg", type=float, default=0.1,
                    help="weight of entropy term in actor loss")                    

if __name__ == "__main__":
    args = parser.parse_args()
    roll_len = args.rollout_len
    actor_lr = args.actor_lr
    critic_lr = args.critic_lr
    entropy_reg = args.entropy_reg
    single_train(roll_len, actor_lr, critic_lr, entropy_reg)            
            

