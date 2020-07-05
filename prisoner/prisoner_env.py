import torch
import random
import numpy as np

# each transition is a list with the format of [s,a,s',r]

# Replay buffer from pytorch implementaion
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = args
        self.position = (self.position + 1) % self.capacity

    def add_reward(self, reward):
        curr_index = (self.position - 1) % self.capacity
        self.memory[curr_index][-1] = reward

    def sample(self, batch_size):
        if batch_size > self.capacity:
            return self.memory
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)          
'''
# Another definition for replay memory that is more efficient
# Code adapted from https://github.com/sfujim/TD3/blob/master/utils.py
class ReplayMemory:
    '''
    A replay buffer, if select_phase is False, then only put the state transitions
    during game playing phase into the buffer.
    Otherwise, put the state trainsitions in partner selection phase into the buffer  
    '''
    def __init__(self, h, action_dim, num_agents, capacity):    
        self.ptr_select_phase = 0
        self.ptr_game_phase = 0
        self.capacity = capacity
        state_dim1 = 2*h
        state_dim2 = 2*h*(num_agents-1)
        self.size = 0
        self.selection_state = np.zeros((capacity, state_dim2))
        self.selection_action = np.zeros((capacity, num_agents))
        self.selection_reward = np.zeros((capacity, 1))
        self.game_state = np.zeros((capacity, state_dim1))
        self.game_action = np.zeros((capacity, action_dim))
        self.game_next_state = np.zeros((capacity, state_dim1))
        self.reward = np.zeros((capacity, 1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, select_phase=False):
        if select_phase:
            self.selection_state[self.ptr_select_phase] = state
            if isinstance(action, torch.Tensor):
                action = action.detach().numpy()
                self.selection_action[self.ptr_select_phase] = action
            self.selection_reward[self.ptr_select_phase] = np.reshape(reward,(1,1))
            self.ptr_select_phase = (self.ptr_select_phase + 1) % self.capacity
        else:  
            self.game_state[self.ptr_game_phase] = state 
            if isinstance(action, torch.Tensor): 
                action = action.detach().numpy()
                self.game_action[self.ptr_game_phase] = action
            self.game_next_state[self.ptr_game_phase] = next_state
            self.reward[self.ptr_game_phase] = np.reshape(reward,(1,1))
            self.ptr_game_phase = (self.ptr_game_phase + 1) % self.capacity

        self.size = min(self.size + 1/2, self.capacity)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.selection_state[ind]),
            torch.FloatTensor(self.selection_action[ind]),
            torch.FloatTensor(self.selection_reward[ind]),
            torch.FloatTensor(self.game_state[ind]),
            torch.FloatTensor(self.game_action[ind]),
            torch.FloatTensor(self.game_next_state[ind]),
            torch.FloatTensor(self.reward[ind])
        )

class Agent:
    def __init__(self, action, capacity, h, action_dim, num_agents):
        self.action = action
        self.h = h
        # Memory is only used to remember last step actions
        # Memory has length equal to h
        self.memory = []
        # Replay is used to store the trainsitions(s,a,s',r,done)
        self.replay = ReplayMemory(h, action_dim, num_agents, capacity) 
    
    def remember(self, action):
        # Remember the last action taken
        if not isinstance(action, list):
            action = action.tolist()
        self.memory.append(action)
        if len(self.memory) > self.h:
            self.memory.pop(0)

    def state(self):
        # construct the input states, which has shape [h,2]       
        return torch.tensor(self.memory)

class PrisonerEnv:
    def __init__(self, agent1, agent2, reward):
        # Reward is a type of dictionary
        self.agent1 = agent1
        self.agent2 = agent2
        self.round_reward = -1
        self.reward_matrix = reward

    def get_action(self):
        return [self.agent1.action, self.agent2.action]

    def set_action(self, ac_1, ac_2):
        self.agent1.action = ac_1
        self.agent2.action = ac_2    

    def set_agents(self, ag1, ag2):
        self.agent1 = ag1
        self.agent2 = ag2     

    def step(self):
        ac_1 = self.agent1.action
        ac_2 = self.agent2.action
        self.round_reward = self.reward_matrix[(ac_1,ac_2)]
        return self.round_reward

    def __repr__(self):
        str1 = "The action of agent 1 is:" + str(self.agent1.action)
        str2 = "The action of agent 2 is:"+ str(self.agent2.action)
        str3 = "The corresponding reward is:" +  str(self.round_reward)
        return str1+'\n'+str2+'\n'+str3    

# test sample usage
'''
def main():
    # 0 - cooperate, 1-defect
    agent1 = Agent(0,5,1)
    agent2 = Agent(1,5,1)
    reward = {(0,0):[3,3],(0,1):[0,4],(1,0):[4,0],(1,1):[1,1]}
    env = PrisonerEnv(agent1, agent2, reward)
    env.step()  
    print(repr(env))     

if __name__ == "__main__":
    main()    
'''    