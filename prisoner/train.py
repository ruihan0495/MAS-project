import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
from itertools import count
from dqn import DQN
from model import GNNSelect
from prisoner_env import Agent, PrisonerEnv
import argparse

NUM_AGENTS = 4
BATCH_SIZE = 16
CONT_PROP = 0.95  
GAMMA = 0.99

class CoumpoundNet(nn.Module):
    def __init__(self, num_agents, num_actions, num_time_steps, hidden_dim1, hidden_dim2):
        super(CoumpoundNet, self).__init__()
        self.qnet = DQN(num_agents, num_actions, hidden_dim1, num_time_steps)
        self.gnn = GNNSelect(num_agents, num_time_steps, hidden_dim2, BATCH_SIZE)
   

'''a function to compute input for gnn when h=1'''
def concate_states(actions):
    #print('Actions:',actions)
    num_agents = len(actions)
    all_states = torch.zeros(num_agents,num_agents-1,2)
    with torch.no_grad():
        for i in range(num_agents):
            if i == 0:
                ns_i = actions[1:]
            elif i == num_agents-1:
                ns_i = actions[0:-1]
            else:
                ns_i = actions[0:i]+actions[i+1:]
            temp = torch.Tensor(num_agents-1,2)
            #print('first ns_i:',ns_i)
            ns_i = torch.cat(ns_i, out=temp).reshape(temp.shape) 
            #print(ns_i)
            #print(all_states.shape, ns_i.shape)
            all_states[i,:,:] = ns_i.squeeze()
        return all_states

# TODO: Define another function that similar to concate_state but record h steps    



def train(model, lr, env, num_episodes,agents, h=1):
    # For plotting purposes
    total_coop = []
    total_mutual_coop = []
    total_mutual_deft = []
    total_deft = []

    optimizers = [None] * len(agents)
    for i in range(len(agents)):

        optimizers[i] = torch.optim.Adam(model[i].parameters(), lr)

    # Agents is a list of agent
    for j in range(num_episodes):
        rand_num = random.random()
        #counter = 0
        if rand_num < CONT_PROP:
            action = []
            #partners = []
            for i in range(NUM_AGENTS):
                # Collect past h actions
                curr_agent = agents[i]
                #if counter < h:
                action_n = nn.functional.one_hot(torch.randint(2,(1,)), num_classes=2)
                curr_agent.remember(action_n)
                s_i= curr_agent.state()
                # Convert to type float
                s_i = s_i.type(torch.FloatTensor)
                #print('input shape is:', s_i.shape)
                q_val = model[i].qnet(s_i) 
                a_d = model[i].qnet.select_action(q_val)
                #print('output shape of a_d:', a_d.shape)
                curr_agent.remember(a_d)
                next_s = curr_agent.state()
                action.append(a_d)
                # Add s_i, a_d, s_i' to replay buffer
                curr_agent.replay.push([s_i,a_d,next_s,None])
            all_states = concate_states(action)  

            num_coop = 0
            num_mutual_coop = 0
            num_mutual_deft = 0
            num_deft = 0

            for i in range(NUM_AGENTS):
                # Collect states for all other agents excludes itself
                # ns_i has shape [num_agents, num_agents-1, 2, h=1]
                ns_i = all_states.unsqueeze(-1)
                rel = torch.empty(4, 4).random_(2)
                logits = model[i].gnn(rel, ns_i)
                #print('logits has shape:', logits.shape)
                interval = model[i].gnn.num_agents-1
                a_s = F.gumbel_softmax(logits=logits[i*interval:(i+1)*interval,:], hard=True, dim=1) 
                partner = agents[torch.argmax(a_s)]
                env.set_agents(agents[i],partner)
                reward = env.step()
                agents[i].replay.add_reward(torch.tensor(reward))
                #print('the replay buffer is:', np.array(agents[i].replay.sample(BATCH_SIZE)).shape)
                batch_replay = np.array(agents[i].replay.sample(BATCH_SIZE))
                #print(batch_replay[:,-1])
                batched_next_s, batched_reward = torch.stack(list(batch_replay[:,-2])), torch.stack(list(batch_replay[:,-1]))
                batched_next_s = batched_next_s.reshape(-1,1,2)

                if agents[i].memory[-1]==[[[1,0]]]:
                    num_coop += 1
                if agents[i].memory[-1]==[[[1,0]]] and partner.memory[-1]==[[[1,0]]]:
                    num_mutual_coop += 1
                elif agents[i].memory[-1]==[[[0,1]]] and partner.memory[-1]==[[[0,1]]]:
                    num_mutual_deft += 1
                else:
                    num_deft += 1 

                #print(batched_next_s, batched_next_s.shape, batched_reward)
                target = batched_reward + GAMMA*torch.max(model[i].qnet(batched_next_s))
                batched_s = torch.stack(list(batch_replay[:,0]))
                curr_qval = model[i].qnet(batched_s).squeeze()
                loss = F.mse_loss(curr_qval, target)
                optimizers[i].zero_grad()
                loss.backward(retain_graph=True)
                optimizers[i].step()
        
        total_coop.append(num_coop)
        total_mutual_coop.append(num_mutual_coop)
        total_mutual_deft.append(num_mutual_deft)
        total_deft.append(num_deft)

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
    plt.savefig('figures\plot2.png')

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
    #NUM_AGENTS = args.num_agents
    #BATCH_SIZE = args.batch_size
    #CONT_PROP = args.cont_prop   
    #GAMMA = args.gamma
    #EPS_START = 0.9
    #EPS_END = 0.05
    #EPS_DECAY = 200


  
    agent1 = Agent(0,5,1)
    agent2 = Agent(1,5,1)
    agent3 = Agent(0,5,1)
    agent4 = Agent(1,5,1)

    agents = [agent1, agent2, agent3, agent4]
    model = []
    reward = {(0,0):[3,3],(0,1):[0,4],(1,0):[4,0],(1,1):[1,1]}
    env = PrisonerEnv(agent1, agent2, reward)
    for i in range(len(agents)):
        model.append(CoumpoundNet(NUM_AGENTS, 2, 1, 32, 16))
    train(model, lr, env, 2500, agents) 
    print('Done!')