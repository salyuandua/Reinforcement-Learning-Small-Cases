# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 10:54:01 2018
Training computer to play Taxi-v2 with Q table 
The game enviroment provided by OpenAI gym lib
@author: Xuejian Li
"""
import numpy as np
import gym
import random

#get game environment
taxi_env=gym.make("Taxi-v2")
taxi_env.render()

taxi_state=taxi_env.observation_space.n
taxi_action=taxi_env.action_space.n

#create q table
q_table=np.zeros((taxi_state,taxi_action))

#init hyperparamters
total_episodes = 15000        # Total episodes
learning_rate = 0.8           # Learning rate
max_steps = 99                # Max steps per episode
gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.005             # Exponential decay rate for exploration prob

#Q learning
for episode in range(total_episodes):
    #get init state randomly
    init_state=taxi_env.reset()
    done=False
    #run game
    for step in range(max_steps):
        tradeoff=random.uniform(0,1)
        
        if tradeoff>episode:
            action=np.argmax(q_table[init_state,:])
        else:
            action=taxi_env.action_space.sample()
        new_state, reward, done, info=taxi_env.step(action)    
        q_table[init_state,action]=q_table[init_state,action]+learning_rate*(reward+gamma*np.max(q_table[new_state,:])-q_table[init_state,action])
        #update state
        init_state=new_state
        #the game is done
        if done==True:
            break
    episode+=1
     #reduce episode
    episode=min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
     
#play 
taxi_env.reset()
total_rewards=[]#reward
total_test_episode=100
for episode in range(total_test_episode):
    state=taxi_env.reset()
    done=False
    rewards=0
    for step in range(max_steps):
        #pick an action
        action=np.argmax(q_table[state,:])
        #go next step
        new_state, reward, done, info=taxi_env.step(action)
        #sum rewards
        rewards=rewards+reward
        #if done, break
        if done:
            total_rewards.append(rewards)
            print("In "+str(episode)+"th episode, reward is "+str(rewards))
            break
        state=new_state
taxi_env.close()        
        
        
    
    






     
     
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    








