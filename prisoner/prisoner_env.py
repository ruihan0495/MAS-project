import torch
import random

# each transition is a list with the format of [s,a,s',r]

# Replay buffer from pytorch implementaion
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
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



class Agent:
    def __init__(self, action, capacity, h):
        self.action = action
        self.h = h
        self.memory = []
        self.replay = ReplayMemory(capacity)
    
    def remember(self, action):
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