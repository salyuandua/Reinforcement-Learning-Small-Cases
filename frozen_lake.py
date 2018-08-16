# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 17:02:51 2018

@author: Xuejian Li
"""
import numpy as np
import gym
import random

#get environment
lake_env=gym.make("FrozenLake-v0")
#get sizes of actions and states
action_size=lake_env.action_space.n
state_size=lake_env.observation_space.n
#init Q table
q_table=np.zeros((state_size,action_size))
#define some hyperparamters
total_episodes=10000 #total number of iterations of training
learning_rate=0.9    #learning rate
max_step=100         #max number of steps per episode
gamma=0.9            #discounted rate
total_rewards=[]     #buffer of rewards
epsilon = 1.0        # Exploration rate
max_epsilon = 1.0    # Exploration probability at start
min_epsilon = 0.01   # Minimum exploration probability 
decay_rate = 0.005 

#training process
for episode in range(total_episodes):
           
            
    #init environment
    init_state=lake_env.reset()
    rewards=0
    done=False
    #to play
    for step in range(max_step):
        #get a random number whose mean is 0 and deviation is 1
        tradeoff=random.uniform(0,1)
        if tradeoff>episode:
            action=np.argmax(q_table[init_state,:])
        else:#take next action randomly
            action=lake_env.action_space.sample()
        new_state, reward, done, info=lake_env.step(action)
        #Q(s,a):= Q(s,a) + learning_rate * [R(s,a) + discounted_rate * max Q(s',a') - Q(s,a)]
        q_table[init_state,action]=q_table[init_state,action]+\
        learning_rate*(reward+gamma*np.max(q_table[new_state,:])-q_table[init_state,action])
        #update rewards
        rewards+=reward
        #update state
        init_state=new_state
        if done:
            break
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)       
    total_rewards.append(rewards)        
            
print(q_table)            
        
#to play game with Q table
#init environment
lake_env.reset()
total_test_episode=100
for episode in range(total_test_episode):
    init_state=lake_env.reset()
    done=False
    rewards=0
    for step in range(max_step):
        action=np.argmax(q_table[init_state,:])
        new_state, reward, done, info = lake_env.step(action)
        if done:
            




        
        
        
        
        
        
        
    
    
    
    








 
















