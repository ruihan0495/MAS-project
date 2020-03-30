import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

'''
A game engine for minority game:
step - a function called on the environment and return the next state, reward
reward - a function calculate reward for agents
state - a function generates the current state for agent
'''
class MinorGameEnv(gym.Env):
    def __init__(self, k):
        self.k = k
        self.num_agents = 2*k+1
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1,
                                 shape=(self.num_agents,2),
                                 dtype=np.float32)
        self.action_list = np.array([0] * self.num_agents)
        self.reward_list = np.array([0] * self.num_agents)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_reward(self, agent_id):
        agent_action = self.action_list[agent_id]
        num_same_action = np.count_nonzero(self.action_list==agent_action)
        if num_same_action <= self.k:
            self.reward_list[agent_id] = 1
        else:
            self.reward_list[agent_id] = 0

    def get_observation(self):
        obs = np.zeros((self.num_agents,2),dtype=np.float32)
        for i in range(self.num_agents):
            obs[index,:] = [self.action_list[i],self.reward_list[i]]
        return obs

    def step(self, action_n):
        # update actions together
        for agent_id in range(self.num_agents):
            self.action_list[agent_id] = action_n[agent_id]
        # update rewards together    
        for agent_id in range(self.num_agents):    
            self.set_reward(agent_id)    
        next_observation = self.get_observation()
        reward_list = self.reward_list
        return reward_list, next_observation

    def reset(self):
        self.action_list = np.array([0] * self.num_agents)
        self.reward_list = np.array([0] * self.num_agents)

    def render(self):
        # print the action and reward list at current stage
        print("Current actions are: ", self.action_list)
        print("Current rewards are: ", self.reward_list)