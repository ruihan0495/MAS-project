import torch
import torch.nn as nn
import random
import numpy as np
from collections import namedtuple
from itertools import count
from dqn import DQN, ReplayMemory
from model import GNNSelect
import argparse

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
CONT_PROP = 0.95 #probability of continue play game 
NUM_AGENTS = 4


#TODO: define model, compound net with GNN+Q_NET

def single_train(model, env, num_episodes,agents, h=1):
    # agents is a list of agent
    for j in range(num_episodes):
        rand_num = random.random()
        #counter = 0
        if rand_num < CONT_PROP:
            action = []
            partners = []
            for i in range(NUM_AGENTS):
                # collect past h actions
                curr_agent = agents[i]
                #if counter < h:
                action_n = nn.functional.one_hot(torch.randint(2,(3,)), num_classes=2)
                curr_agent.remember(action_n)
                s_i= curr_agent.state()
                a_d = model.qnet(s_i) # assume a_d is one_hot??? need to check how to cast float tensor to one-hot tensor while differentiable
                curr_agent.remember(a_d)
                next_s = curr_agent.state()
                action.append(a_d)
                # add s_i, a_d, s_i' to replay buffer
                curr_agent.replay.push((s_i,a_d,next_s))







'''
def single_train(model, env, rollout_length, actor_lr, critic_lr, entropy_reg, gamma=0.2):    
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
            print("The predicted policy distribution is: {}".format(dist.data))
            # sample next action
            next_action = model.select_action(env.action_space, dist)
            env.action_list[0] = next_action
            rewards, state_i = env.step(env.action_list)
            # state_i has shape[num_agents, num_actions]
            hor_state_i = np.reshape(state_i,(1,-1))
            temp_aug_state = np.vstack((aug_state, hor_state_i))
            temp_aug_state = temp_aug_state[1:,:]
            input = torch.tensor(temp_aug_state)
            _, next_value = model(input)
            # train network
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
 
            log_probs = torch.sum(torch.log(dist))
            advantange = torch.tensor(rewards) +  gamma*next_value - value
            entropy = torch.sum(-torch.mean(dist)*torch.log(dist))
            actor_loss =torch.mean(log_probs*advantange) + entropy_reg*entropy            
            critic_loss = loss_fn(torch.tensor(rewards).unsqueeze(1)+next_value, value.view(-1,1))
    
            actor_loss.backward(retain_graph=True)
            critic_loss.backward()
            actor_optimizer.step()
            critic_optimizer.step()


def multi_train(model, env, num_agents, rollout_length, actor_lr, critic_lr, entropy_reg, gamma=0.2):
    state_i = env.get_observation()   
    loss_fn = nn.MSELoss()
    states_so_far = deque()
    aug_states_so_far = deque()
    # instantiate a list of optimizers
    actor_optimizers = [0]*num_agents
    critic_optimizers = [0]*num_agents
    actor_losses = [0]*num_agents
    critic_losses = [0]*num_agents
    dists=[]
    values=[]
    #total_r = []
    for i in range(num_agents):
        actor_optimizers[i] = torch.optim.Adam(model[i].parameters(),lr=actor_lr)
        critic_optimizers[i] = torch.optim.Adam(model[i].parameters(),lr=critic_lr)
    for i in range(rollout_length):
        states_so_far.append(state_i)
        if i < model[0].num_time_steps:
            action_n = utils.random_action(model[0].num_agents)
            rewards, state_i = env.step(action_n)
            #total_r.append(rewards)
            print(len(states_so_far))   
        # concat frames in past k time steps
        else:
            j = 1
            aug_state = states_so_far[i]
            while j < model[0].num_time_steps:
                aug_state = np.hstack((aug_state, states_so_far[i-j]))
                j+=1
            aug_states_so_far.append(aug_state)
            # feed into network
            input = torch.tensor(aug_state[np.newaxis,:,:])
            for i in range(num_agents):
                model_i = model[i]           
                dist, value = model_i(input)
                dists.append(dist)
                values.append(value)
                print("The predicted policy distribution is: {}".format(dist.data))
                # sample next action
                next_action = model_i.select_action(env.action_space, dist)
                env.action_list[i] = next_action
            rewards, state_i = env.step(env.action_list)
            #total_r.append(rewards)
            # state_i has shape[num_agents, num_actions]
            hor_state_i = np.reshape(state_i,(1,-1))
            temp_aug_state = np.vstack((aug_state, hor_state_i))
            temp_aug_state = temp_aug_state[1:,:]
            input = torch.tensor(temp_aug_state)
            for i in range(num_agents):
                model_i = model[i]
                _, next_value = model_i(input)
                # train network
                actor_optimizers[i].zero_grad()
                critic_optimizers[i].zero_grad()
    
                log_probs = torch.sum(torch.log(dists[i]))
                advantange = torch.tensor(rewards) +  gamma*next_value - values[i]
                entropy = torch.sum(-torch.mean(dists[i])*torch.log(dists[i]))
                actor_losses[i] =torch.mean(log_probs*advantange) + entropy_reg*entropy            
                critic_losses[i] = loss_fn(torch.tensor(rewards).unsqueeze(1)+next_value, values[i].view(-1,1))
                actor_losses[i].backward(retain_graph=True)
                critic_losses[i].backward()
                actor_optimizers[i].step()
                critic_optimizers[i].step()
'''

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
    env = minority_game.MinorGameEnv(2)
    num_agents = 5
    model = A2C(2,6,2,5) 
    models = [model]*num_agents
    #single_train(model, env, roll_len, actor_lr, critic_lr, entropy_reg)
    multi_train(models, env, 5, roll_len, actor_lr, critic_lr, entropy_reg)  