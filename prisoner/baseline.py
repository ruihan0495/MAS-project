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
import copy

# This script implement the algorithm in 
# https://www.aaai.org/Papers/AAAI/2020GB/AAAI-AnastassacosN.1598.pdf
# as a baseline approach 

NUM_AGENTS = 4
BATCH_SIZE = 64
CONT_PROP = 0.95  
GAMMA = 0.99
T = 50
TAU = 0.005

def states_for_i(actions, i):
    '''
    Args:
        actions[list] - a list of actions executed for all agents at last time step
        i[int] - the index of agent
    Return:
        the constructed state representation that has shape [1, 2h*(num_agents-1)]
        for the selection Q-network. This state is the concatenation of the last time step 
        actions of all other agents exclude agent i    
    '''
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

def onehot_to_int(action1, action2):
    '''
    Args:
        action1, action2[(differentiable) tensor] - tensors of actions taken
        by agent 1 and agent2 in one-hot format, for example, action1=
        tensor([[0.,1.]], grad_fn=..) or action1=tensor([[0.,1.]])
    Return:
        int version of action1 and action2, that is, tensor([[0.,1.]]) = 1 and
        tensor([[1.,0.]]) = 0    
    '''
    action1 = action1.detach().numpy().argmax()
    action2 = action2.detach().numpy().argmax()
    return action1, action2


class CoumpoundNet(nn.Module):
    '''
    Define a compound network that has 2 sub-networks:
        A qnet for game playing Q-learning
        A q_select for partner selection Q-learning
        A qnet_target to stablize training
    '''
    def __init__(self, num_agents, num_actions, num_time_steps, hidden_dim1, hidden_dim2):
        super(CoumpoundNet, self).__init__()
        self.qnet = DQN(num_agents, num_actions, hidden_dim1, num_time_steps)
        # Target network for game playing Q-net
        self.qnet_target = copy.deepcopy(self.qnet)
        self.q_select = DQNSelect(num_agents, num_time_steps, hidden_dim2)       

def train(model, game_lr, select_lr, env, num_episodes, agents, update_freq, h=1):
    '''
    Args:
        model[list of CompoundNet] - a list of models, each agent has it's own decision making
        model
        lr[float] - learning rate
        agents[list of agent] - a list of agents
        update_freq[int] - the frequency to update qnet_target in training

    The training schema is:
        1. each agent randomly picks action at the very begining until it has enough history
        to construct game playing state representations, in this case, at least h=1 actions must be taken
        2. feed the game playing state representation into qnet to select next action for each agent
        3. construct partner selection state representations for each agent using states_for_i
        4. feed the partner selection state representation into q_select to get the partner to play with
        5. the reward is obtained via the interaction of agent and it's selected partner in the environment   
    '''
    # Variables for plotting purpose
    total_coop = []
    total_mutual_coop = []
    total_mutual_deft = []
    total_exploitation = []
    total_deception = []
    total_reward = []

    num_coop = 0
    num_mutual_coop = 0
    num_mutual_deft = 0
    num_deception = 0
    num_exploitation = 0 
    avg_reward = 0  

    # Initialize optimizers for q_net and q_select, in this case, all Adam
    game_optimizers = [None] * len(agents)
    partner_optimizers = [None] * len(agents)
    for i in range(len(agents)):
        game_optimizers[i] = torch.optim.Adam(model[i].qnet.parameters(), game_lr)
        partner_optimizers[i] = torch.optim.Adam(model[i].q_select.parameters(), select_lr)
    for j in range(num_episodes):
        rand_num = random.random()
        round_count = 0
        # An indicator to count all many random actions for each agent have been picked at beginning
        init_actions = [0] * NUM_AGENTS

        while rand_num < CONT_PROP:
            round_count += 1
            
            num_coop = 0
            num_mutual_coop = 0
            num_mutual_deft = 0
            num_deception = 0
            num_exploitation = 0 
            avg_reward = 0   

            prev_action = []
            action = []
            round_states = []
            round_next_states = []         

            for i in range(NUM_AGENTS):
                 
                # This part encode the game playing phase
                # Collect past h actions
                init_actions[i] += 1
                curr_agent = agents[i]
                # If initially less then h, act randomly; else the next action is picked by qnet with epsilon-greedy policy
                if init_actions[i] <= 1:
                    action_n = F.one_hot(torch.randint(2,(1,)), num_classes=2).type(torch.FloatTensor)
                else:
                    action_n = a_d

                # Each agent remember the last action taken and construct state representation for qnet
                curr_agent.remember(action_n)
                s_i = curr_agent.state()
                # Convert to type float
                s_i = s_i.type(torch.FloatTensor).reshape(1,-1)
      
                q_val = model[i].qnet(s_i) 
                a_d = model[i].qnet.select_action(q_val)
                # Store next states and next actions for each agent
                next_s = a_d.reshape(1,-1)
                #prev_action.append(action_n)
                prev_action.append(action_n)
                action.append(a_d)
                round_states.append(s_i)
                round_next_states.append(next_s)            

            for i in range(NUM_AGENTS):
                # Collect states for all other agents excludes itself
                # ns_i has shape [1, (num_agents-1)*2*h]
                ns_i = states_for_i(prev_action, i)
                logits = model[i].q_select(ns_i)
                # Partner selection
                partner_id, a_s = model[i].q_select.select_partner(i, logits)
                partner = agents[partner_id]
                # Agent i and it's selected partner play against each other in the environment
                env.set_agents(agents[i], partner)
                act1, act2 = onehot_to_int(prev_action[i], prev_action[partner_id])
                env.set_action(act1, act2)           
                s_reward = env.step()

                act1, act2 = onehot_to_int(action[i], action[partner_id])
                env.set_action(act1, act2)           
                g_reward = env.step()
                # Add game playing histories s, a, s', r to the replay buffer 
                agents[i].replay.add(round_states[i], action[i], round_next_states[i], g_reward[0])
                # Add partner selction histories s, a to the replay buffer
                agents[i].replay.add(ns_i, a_s, 0, s_reward[0], select_phase=True)
            
                s_state, s_action, _, g_state, g_action, g_next_state, g_reward= agents[i].replay.sample(BATCH_SIZE)
                g_action = torch.argmax(g_action, dim=1)   
                s_action = torch.argmax(s_action, dim=1)
                #print('agent {}'.format(i), agents[i].memory)              
                if agents[i].memory[-1]== [[1.0,0.0]]:
                    num_coop += 1

                if agents[i].memory[-1]== [[1.0,0.0]] and partner.memory[-1]== [[1.0,0.0]]:
                    num_mutual_coop += 1
                    #env.set_agents(agents[i],partner)
                    env.set_action(0,0)
                    avg_reward += env.step()[0]
                elif agents[i].memory[-1]== [[0.0,1.0]] and partner.memory[-1]== [[0.0,1.0]]:
                    num_mutual_deft += 1
                    #env.set_agents(agents[i],partner)
                    env.set_action(1,1)
                    avg_reward += env.step()[0]
                elif agents[i].memory[-1]== [[0.0,1.0]] and partner.memory[-1]== [[1.0,0.0]]:
                    num_exploitation += 1
                    env.set_action(1,0)
                    avg_reward += env.step()[0]
                else:
                    num_deception += 1 
                    env.set_action(0,1)
                    avg_reward += env.step()[0]

                # Training Q-net for game playing phase
                with torch.no_grad():
                    target = g_reward+ GAMMA*torch.max(model[i].qnet_target(g_next_state), dim=1, keepdim=True)[0]   
                curr_qval = model[i].qnet(g_state).gather(1,g_action.unsqueeze(1))
                loss = F.mse_loss(curr_qval, target)
                game_optimizers[i].zero_grad()
                loss.backward()
                game_optimizers[i].step()                

                # Training Q-select for partner selection phase
                with torch.no_grad():
                    target = g_reward+ GAMMA*torch.max(model[i].qnet_target(g_state), dim=1, keepdim=True)[0]
                curr_qval = model[i].q_select(s_state).gather(1,s_action.unsqueeze(1))
                loss = F.mse_loss(curr_qval, target)
                partner_optimizers[i].zero_grad()
                loss.backward()
                partner_optimizers[i].step()


                # Polyak target average
                for param, target_param in zip(model[i].qnet.parameters(), model[i].qnet_target.parameters()):
                    target_param.data.copy_(TAU * param.data + (1-TAU) * target_param.data)
            
            if round_count >= T:
                break
        
        normalizer = NUM_AGENTS
        total_coop.append(num_coop/normalizer)
        total_mutual_coop.append(num_mutual_coop/normalizer)
        total_mutual_deft.append(num_mutual_deft/normalizer)
        total_exploitation.append(num_exploitation/normalizer)
        total_deception.append(num_deception/normalizer)
        total_reward.append(avg_reward/normalizer)

    # Plotting  
    plot1 = plt.figure(1)  
    plt.plot(total_coop, alpha=0.5)
    plt.xlabel('iterations')
    plt.ylabel('cooperative agents')
    plt.savefig('figures\plot1.png')

    plot2 = plt.figure(2)
    plt.plot(total_mutual_coop, color = 'red', alpha = 0.5, label='mutual_coop')
    plt.plot(total_mutual_deft, color = 'green', alpha = 0.5, label='mutual_defect')
    plt.plot(total_deception, color='blue', alpha = 0.5, label='deception')
    plt.plot(total_exploitation, color='orange', alpha = 0.5, label='exploitation')
    #plt.plot(total_reward, color='purple', alpha = 0.5, label='total reward')
    plt.ylabel('interactions')
    plt.xlabel('iterations')
    plt.legend()
    plt.savefig('figures\plot2.png')
        
      



parser = argparse.ArgumentParser()

## General parameters
parser.add_argument("--game_lr", type=float, default=0.01,
                    help="game playing learning rate ")

parser.add_argument("--select_lr", type=float, default=0.001,
                    help="partner selection learning rate ")                    

parser.add_argument("--update_freq", type=int, default=15,
                    help="number of iterations to update the policy network") 

parser.add_argument("--replay_capacity", type=int, default=100,
                    help="the size of agent's replay buffer")                    

#parser.add_argument("batch_size", type=int, default=16,
#                    help="the training batch size")  

#parser.add_argument("gamma", type=float, default=0.99,
#                    help="the discount rate when calculate the q_net target")

#parser.add_argument("cont_prop", type=float, default=0.95,
#                    help="the probability of continue play game in each epsisode")            

                                                           


if __name__ == "__main__":
    args = parser.parse_args()
    game_lr = args.game_lr
    select_lr = args.select_lr
    update_freq = args.update_freq
    replay_capacity = args.replay_capacity
    actions = [0, 1]
    agents = []
    for i in range(NUM_AGENTS):
        action = random.choice(actions)
        agent = Agent(action,replay_capacity,1,2,NUM_AGENTS)
        agents.append(agent)
    models = []
    reward = {(0,0):[3,3],(0,1):[0,4],(1,0):[4,0],(1,1):[1,1]}
   
    # Random pick 2 different agents to start the game
    agent1 = random.choice(agents)
    agent2 = agent1
    while agent2 == agent1:
        agent2 = random.choice(agents) 

    env = PrisonerEnv(agent1, agent2, reward)
    for i in range(len(agents)):
        models.append(CoumpoundNet(NUM_AGENTS, 2, 1, 256, 256))
    train(models, game_lr, select_lr, env, 100, agents, update_freq) 
    print('Done!')   