import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
from itertools import count
from dqn import DQN
from model import DQNSelect
from prisoner_env import Agent, PrisonerEnv
import argparse

# This script implement the algorithm in 
# https://www.aaai.org/Papers/AAAI/2020GB/AAAI-AnastassacosN.1598.pdf
# as a baseline approach

NUM_AGENTS = 4
BATCH_SIZE = 16
CONT_PROP = 0.95  
GAMMA = 0.99
T = 10

def states_for_i(actions, i):
    #print('Actions:',actions)
    num_agents = len(actions)
    with torch.no_grad():
        if i == 0:
            ns_i = actions[1:]
        elif i == num_agents-1:
            ns_i = actions[0:-1]
        else:
            ns_i = actions[0:i]+actions[i+1:]
        temp = torch.Tensor(num_agents-1,2)
        ns_i = torch.cat(ns_i, out=temp).reshape(temp.shape) 
        ns_i = torch.flatten(ns_i).reshape(1,-1) 
        #print('ns_i',ns_i)
        return ns_i

class CoumpoundNet(nn.Module):
    def __init__(self, num_agents, num_actions, num_time_steps, hidden_dim1, hidden_dim2):
        super(CoumpoundNet, self).__init__()
        self.qnet = DQN(num_agents, num_actions, hidden_dim1, num_time_steps)
        self.q_select = DQNSelect(num_agents, num_time_steps, hidden_dim2)

def train(model, lr, env, num_episodes, agents, h=1):
    optimizers = [None] * len(agents)
    for i in range(len(agents)):
        optimizers[i] = torch.optim.Adam(model[i].parameters(), lr)

    for j in range(num_episodes):
        rand_num = random.random()
        round_count = 0
        while rand_num < CONT_PROP:
            round_count += 1
            action = []
            round_states = []
            round_next_states = []
            for i in range(NUM_AGENTS):
                # This part encode the partner selection phase
                # Collect past h actions
                curr_agent = agents[i]
                action_n = F.one_hot(torch.randint(2,(1,)), num_classes=2)
                curr_agent.remember(action_n)
                s_i = curr_agent.state()
                # Convert to type float
                s_i = s_i.type(torch.FloatTensor).reshape(1,-1)
      
                q_val = model[i].qnet(s_i) 
                a_d = model[i].qnet.select_action(q_val)
   
                # Store states and actions for agents
                curr_agent.remember(a_d)
                next_s = curr_agent.state().type(torch.FloatTensor).reshape(1,-1)
                             
                action.append(a_d)
                round_states.append(s_i)
                round_next_states.append(next_s) 
                 

            """ num_coop = 0
            num_mutual_coop = 0
            num_mutual_deft = 0
            num_deft = 0 """

            for i in range(NUM_AGENTS):
                # Collect states for all other agents excludes itself
                # ns_i has shape [1, (num_agents-1)*2*h]
                ns_i = states_for_i(action, i)
                logits = model[i].q_select(ns_i)
 
                a_s = F.gumbel_softmax(logits, hard=True, dim=1) 
                partner = agents[torch.argmax(a_s)]
                env.set_agents(agents[i],partner)
                reward = env.step()
                agents[i].replay.add(round_states[i], action[i], round_next_states[i], reward[0])
                agents[i].replay.add(ns_i, a_s, 0, 0, select_phase=True)
            
                s_state, s_action, s_reward, g_state, g_action, g_next_state, g_reward= agents[i].replay.sample(BATCH_SIZE)
               

                """ if agents[i].memory[-1]==[[[1,0]]]:
                    num_coop += 1
                if agents[i].memory[-1]==[[[1,0]]] and partner.memory[-1]==[[[1,0]]]:
                    num_mutual_coop += 1
                elif agents[i].memory[-1]==[[[0,1]]] and partner.memory[-1]==[[[0,1]]]:
                    num_mutual_deft += 1
                else:
                    num_deft += 1 """ 

                #print(batched_next_s, batched_next_s.shape, batched_reward)
                target = g_reward+ GAMMA*torch.max(model[i].qnet(g_next_state))
                curr_qval = model[i].qnet(g_state).gather(1,g_action.type(torch.LongTensor))
                loss = F.mse_loss(curr_qval, target)
                optimizers[i].zero_grad()
                loss.backward(retain_graph=True)
                optimizers[i].step()
            
            if round_count >= T:
                break
        
        """ normalizer = NUM_AGENTS
        total_coop.append(num_coop/normalizer)
        total_mutual_coop.append(num_mutual_coop/normalizer)
        total_mutual_deft.append(num_mutual_deft/normalizer)
        total_deft.append(num_deft/normalizer)

    # Plotting  
    plot1 = plt.figure(1)  
    plt.plot(total_coop)
    plt.xlabel('iterations')
    plt.ylabel('cooperative agents')
    plt.savefig('figures\plot1.png')

    plot2 = plt.figure(2)
    plt.plot(total_mutual_coop, color = 'red', label='mutual_coop')
    plt.plot(total_mutual_deft, color = 'green', label='mutual_defect')
    plt.plot(total_deft, color='blue', label='defect')
    plt.ylabel('interactions')
    plt.xlabel('iterations')
    plt.legend()
    plt.savefig('figures\plot2.png') """




parser = argparse.ArgumentParser()

## General parameters
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate ")

#parser.add_argument("num_agents", type=int, default=4,
#                    help="number of agents in the game") 

#parser.add_argument("batch_size", type=int, default=16,
#                    help="the training batch size")  

#parser.add_argument("gamma", type=float, default=0.99,
#                    help="the discount rate when calculate the q_net target")

#parser.add_argument("cont_prop", type=float, default=0.95,
#                    help="the probability of continue play game in each epsisode")            

                                                           


if __name__ == "__main__":
    args = parser.parse_args()
    lr = args.lr
 
 
    agent1 = Agent(0,5,1,2,NUM_AGENTS)
    agent2 = Agent(1,5,1,2,NUM_AGENTS)
    agent3 = Agent(0,5,1,2,NUM_AGENTS)
    agent4 = Agent(1,5,1,2,NUM_AGENTS)

    agents = [agent1, agent2, agent3, agent4]
    models = []
    reward = {(0,0):[3,3],(0,1):[0,4],(1,0):[4,0],(1,1):[1,1]}
    env = PrisonerEnv(agent1, agent2, reward)
    for i in range(len(agents)):
        models.append(CoumpoundNet(NUM_AGENTS, 2, 1, 32, 16))
    train(models, lr, env, 250, agents) 
    print('Done!')   