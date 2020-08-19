import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from dqn import DQN
from model import DQNSelect
from prisoner_env import Agent, PrisonerEnv
import argparse
import copy

NUM_AGENTS = 20
CONT_PROP = 0.95
T = 10
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.005
'''
This is an experiment: assume we have 19 agents that are either ALLC or ALLD,
and we apply the learning procedure in baseline.py, do we inspect that the agent
learns to play with ALLC partners?
'''

from baseline import CoumpoundNet

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
    total_reward = []
    # Initialize optimizers for q_net and q_select, in this case, all Adam    
    game_optimizer = torch.optim.Adam(model.qnet.parameters(), game_lr)
    partner_optimizer = torch.optim.Adam(model.q_select.parameters(), select_lr)
    for j in range(num_episodes):
        rand_num = random.random()
        round_count = 0
        # An indicator to count all many random actions for each agent have been picked at beginning
        init_actions = [0] 
        action = [0]
        for i in range(1, NUM_AGENTS):
            action.append(agents[i].action) 
        episode_coop = 0
        episode_reward = 0
        while rand_num < CONT_PROP:
            round_count += 1
            rand_num = random.random()
            
            round_states = []
            round_next_states = []         
                            
            # This part encode the game playing phase
            # Collect past h actions
            init_actions[0] += 1
            curr_agent = agents[0]
            # If initially less then h, act randomly; else the next action is picked by qnet with epsilon-greedy policy
            if init_actions[0] <= 1:
                action_n = F.one_hot(torch.randint(2,(1,)), num_classes=2)
            else:
                action_n = a_d

            # Each agent remember the last action taken and construct state representation for qnet
            curr_agent.remember(action_n)
            s_i = curr_agent.state()
            # Convert to type float
            s_i = s_i.type(torch.FloatTensor).reshape(1,-1)
    
            q_val = model.qnet(s_i) 
            a_d = model.qnet.select_action(q_val)
            # Store next states and next actions for each agent
            next_s = a_d.reshape(1,-1)
            action[0] = a_d  
            round_states.append(s_i)
            round_next_states.append(next_s)            

          
            # Collect states for all other agents excludes itself
            # ns_i has shape [1, (num_agents-1)*2*h]
            ns_i = F.one_hot(torch.tensor(action[1:]), num_classes=2).type(torch.FloatTensor).reshape(1,-1)
            
            logits = model.q_select(ns_i)
            # Partner selection
            partner_id, a_s = model.q_select.select_partner(i, logits)                        
            while partner_id == 0:
                partner_id, a_s = model.q_select.select_partner(i, logits)
            partner = agents[partner_id]
            # Agent i and it's selected partner play against each other in the environment
            env.set_agents(agents[0], partner)
            act1 = action[0].detach().numpy().argmax()

            act2 = action[partner_id]
            # If the partner is cooperative
            if act2 == 0:
                episode_coop += 1
            env.set_action(act1, act2)           
            reward = env.step()
            episode_reward += reward[0]
            # Add game playing histories s, a, s', r to the replay buffer 
            agents[0].replay.add(round_states[0], action[0], round_next_states[0], reward[0])
            # Add partner selction histories s, a to the replay buffer
            agents[0].replay.add(ns_i, a_s, 0, 0, select_phase=True)
        
            s_state, s_action, s_reward, g_state, g_action, g_next_state, g_reward= agents[0].replay.sample(BATCH_SIZE)
               

            # Training Q-net for game playing phase
            with torch.no_grad():
                target = g_reward+ GAMMA*torch.max(model.qnet_target(g_next_state), dim=1, keepdim=True)[0]   
            curr_qval = model.qnet(g_state).gather(1,g_action.type(torch.LongTensor))
            loss = F.mse_loss(curr_qval, target)
            game_optimizer.zero_grad()
            loss.backward()
            game_optimizer.step()                

            # Training Q-select for partner selection phase
            with torch.no_grad():
                target = s_reward+ GAMMA*torch.max(model.qnet_target(g_state), dim=1, keepdim=True)[0]
            curr_qval = model.q_select(s_state).gather(1,s_action.type(torch.LongTensor))
            loss = F.mse_loss(curr_qval, target)
            partner_optimizer.zero_grad()
            loss.backward()
            partner_optimizer.step()


            # Polyak target average
            for param, target_param in zip(model.qnet.parameters(), model.qnet_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1-TAU) * target_param.data)
        
            if round_count >= T:
                break
  
        total_reward.append(episode_reward/(round_count+1))
        total_coop.append(episode_coop/(round_count+1))
    
    # Plotting  
    plot_reward = plt.figure(1)  
    plt.plot(total_reward, alpha=0.5)
    plt.xlabel('num_of_episodes')
    plt.ylabel('episodic reward')
    plt.savefig('figures\plot_reward.png')

    plot_coop = plt.figure(2)  
    plt.plot(total_coop, alpha=0.5)
    plt.xlabel('num_of_episodes')
    plt.ylabel('episodic cooperation')
    plt.savefig('figures\plot_coop.png')


def radnom_partner(qnet, env, agents, num_episodes, game_lr):
    game_optimizer = torch.optim.Adam(qnet.parameters(), game_lr)
    qnet_target = copy.deepcopy(qnet)
    curr_agent = agents[0]   
    total_reward = [] 
    total_mutual_coop = []
    total_mutual_deft = [] 
    total_exploitation = []
    total_deception = []
    for j in range(num_episodes):
        num_rand = random.random()
        init_actions = [0] 
        round_count = 0
        episodic_reward = 0
        num_mutual_coop = 0
        num_mutual_deft = 0
        num_exploitation = 0
        num_deception = 0
        while num_rand < CONT_PROP:
            round_count += 1
            round_states = []
            round_next_states = [] 
            init_actions[0] += 1
            # If initially less then h, act randomly; else the next action is picked by qnet with epsilon-greedy policy
            if init_actions[0] <= 1:
                action_n = F.one_hot(torch.randint(2,(1,)), num_classes=2)
            else:
                action_n = a_d
            agent_id = np.random.randint(1, NUM_AGENTS)

            curr_agent.remember(action_n)
            s_i = curr_agent.state()
            # Convert to type float
            s_i = s_i.type(torch.FloatTensor).reshape(1,-1)
    
            q_val = qnet(s_i) 
            a_d = qnet.select_action(q_val)
            # Store next states and next actions for each agent
            next_s = a_d.reshape(1,-1)

            round_states.append(s_i)
            round_next_states.append(next_s) 

            partner = agents[agent_id]
            env.set_agents(curr_agent, partner)
            act1 = a_d.detach().numpy().argmax()
                        
            act2 = agents[agent_id].action
            # If the partner is cooperative
            """ if act2 == 0:
                episode_coop += 1 """

            env.set_action(act1, act2)           
            reward = env.step()
            episodic_reward += reward[0]

            if curr_agent.action == 0 and partner.action == 0:
                num_mutual_coop += 1

            elif curr_agent.action == 1 and partner.action == 1:
                num_mutual_deft += 1

            elif curr_agent.action == 1 and partner.action == 0:
                num_exploitation += 1

            else:
                num_deception += 1 


            # Add game playing histories s, a, s', r to the replay buffer 
            agents[0].replay.add(round_states[0], a_d, round_next_states[0], reward[0])
            # Add partner selction histories s, a to the replay buffer
            agents[0].replay.add(0,0, 0, 0, select_phase=True)
        
            _, _, _, g_state, g_action, g_next_state, g_reward= agents[0].replay.sample(BATCH_SIZE)
            env.set_action(curr_agent.action, partner.action)
            num_rand = random.random()

            with torch.no_grad():
                target = g_reward+ GAMMA*torch.max(qnet_target(g_next_state), dim=1, keepdim=True)[0]   
            curr_qval = qnet(g_state).gather(1,g_action.type(torch.LongTensor))
            loss = F.mse_loss(curr_qval, target)
            game_optimizer.zero_grad()
            loss.backward()
            game_optimizer.step() 

            for param, target_param in zip(qnet_target.parameters(), qnet_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1-TAU) * target_param.data)
                    
            if round_count > T:
                break
        mean_episodic_reward = episodic_reward / (round_count+1)
        normalizer = NUM_AGENTS
        total_reward.append(mean_episodic_reward)  
        total_mutual_coop.append(num_mutual_coop/normalizer)
        total_mutual_deft.append(num_mutual_deft/normalizer)
        total_exploitation.append(num_exploitation/normalizer)
        total_deception.append(num_deception/normalizer)
   
    # Plotting  
    plot_reward = plt.figure(1)  
    plt.plot(total_reward, alpha=0.5)
    plt.xlabel('num_of_episodes')
    plt.ylabel('episodic reward')
    plt.savefig('figures\plot_reward_random.png')

    plot2 = plt.figure(2)
    plt.plot(total_mutual_coop, color = 'red', alpha = 0.5, label='mutual_coop')
    plt.plot(total_mutual_deft, color = 'green', alpha = 0.5, label='mutual_defect')
    plt.plot(total_deception, color='blue', alpha = 0.5, label='deception')
    plt.plot(total_exploitation, color='orange', alpha = 0.5, label='exploitation')
    #plt.plot(total_reward, color='purple', alpha = 0.5, label='total reward')
    plt.ylabel('interactions')
    plt.xlabel('iterations')
    plt.legend()
    plt.savefig('figures\plot_random.png')

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

parser.add_argument("--is_random", type=bool, default=False,
                    help="whether to do random partner selection")                    
               

if __name__ == "__main__":
    args = parser.parse_args()
    game_lr = args.game_lr
    select_lr = args.select_lr
    update_freq = args.update_freq
    replay_capacity = args.replay_capacity
    is_random = args.is_random

    actions = [0, 1]
    agents = []
    for i in range(NUM_AGENTS):
        action = random.choice(actions)
        agent = Agent(action,replay_capacity,1,2,NUM_AGENTS)
        agents.append(agent)

    coop_agents = 0
    for agent in agents:
        if agent.action == 0:
            coop_agents += 1
    print("The number of ALLC agents is: ",coop_agents)            
    
    reward = {(0,0):[3,3],(0,1):[0,4],(1,0):[4,0],(1,1):[1,1]}
    # Random pick 2 different agents to start the game
    agent1 = random.choice(agents)
    agent2 = agent1
    while agent2 == agent1:
        agent2 = random.choice(agents) 

    env = PrisonerEnv(agent1, agent2, reward)

    def run_experiment(is_random):
        if not is_random:
            model = CoumpoundNet(NUM_AGENTS, 2, 1, 256, 256)
            train(model, game_lr, select_lr, env, 1000, agents, update_freq) 
            print('Done!') 
    
        else:
            qnet = DQN(NUM_AGENTS, 2, 256, 1)
            radnom_partner(qnet, env, agents, 1000, game_lr) 
            print('Done!')

    run_experiment(is_random)        


