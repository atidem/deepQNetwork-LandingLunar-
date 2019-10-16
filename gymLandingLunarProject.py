# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 16:31:28 2019

@author: atidem
"""
import os
import gym 
import numpy as np 
from collections import deque
from keras.models import Sequential,load_model
from keras.layers import Dense 
from keras.optimizers import Adam 
import random
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

class Agent:
    def __init__(self,env):
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        
        self.gamma = 0.99
        self.lr = 0.0001
        
        self.epsilon = 1
        self.decay = 0.9993
        self.min_eps = 0.01
        
        self.memory = deque(maxlen=4000)
        
        self.model = self.build_model()
        self.target_model = self.build_model()
    
    def build_model(self):
        model = Sequential()
        model.add(Dense(64,input_dim=self.state_size,activation="relu"))
        model.add(Dense(64,activation="relu"))
        model.add(Dense(self.action_size,activation="linear"))
        model.compile(loss="mse",optimizer=Adam(lr=self.lr))
        return model
    
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
    
    def act(self,state):
        if random.uniform(0,1)<self.epsilon:
            return np.random.choice(self.action_size)
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])
    
    def replay(self,batch_size):
        if len(self.memory)<batch_size:
            return
        minibatch = random.sample(self.memory,batch_size)
        minibatch = np.array(minibatch)
        not_done_indices = np.where(minibatch[:,4]==False)
        y = np.copy(minibatch[:,2])
        
        if len(not_done_indices[0]) > 0:
            predict_sprime = self.model.predict(np.vstack(minibatch[:,3]))
            predict_sprime_target = self.target_model.predict(np.vstack(minibatch[:,3]))
            
            y[not_done_indices] += np.multiply(self.gamma,predict_sprime_target[not_done_indices,np.argmax(predict_sprime[not_done_indices, :][0], axis=1)][0])
        
        actions = np.array(minibatch[:,1],dtype=int)
        y_target = self.model.predict(np.vstack(minibatch[:,0]))
        y_target[range(batch_size),actions] = y
        self.model.fit(np.vstack(minibatch[:,0]),y_target,epochs=1,verbose=0)
        
    """if len(self.memory) < batch_size:
                return
            minibatch = random.sample(self.memory,batch_size)
            for state,action,reward,next_state,done in minibatch:
                if done:
                    target = reward
                else:
                    target = reward + self.gamma*np.amax(self.model.predict(next_state)[0])
                y_train = self.model.predict(state)
                y_train[0][action] = target
                self.model.fit(state,y_train,verbose=0)
     """
    def adaptiveEpsilon(self):
        if self.epsilon > self.min_eps:
            self.epsilon *= self.decay

    def targetModelUpdate(self):
        self.target_model.set_weights(self.model.get_weights())
        
if __name__=="__main__":
    # initialize env. and agent
    env = gym.make("LunarLander-v2")
    agent = Agent(env)
    batch_size = 32   
    episodes = 100
    
    for e in range(episodes):
        #initialize enviroment
        state = env.reset()
        state = np.reshape(state,[1,agent.state_size])
        
        rewardCount = 0 
        for time in range(700):
            #acting , choose a act
            action = agent.act(state)
            #step
            next_state,reward,done,_= env.step(action)
            next_state = np.reshape(next_state,[1,agent.state_size])
            #remember / storage
            agent.remember(state,action,reward,next_state,done)
            #update state
            state = next_state
            #replay
            agent.replay(batch_size)
            
            rewardCount += reward
            
            if done:
                agent.targetModelUpdate()
                break
            
        agent.adaptiveEpsilon()
        print("eps:{} , reward:{}".format(e,rewardCount))
        
# %% test görselleştirme/test visualize
import time 
trained_model = agent
#trained_model.model = load_model("model.h5")
#trained_model.target_model = load_model("target_model.h5")
state = env.reset()
state = np.reshape(state,[1,8])
time_t = 0
while True:
    env.render()
    action = trained_model.act(state)
    next_state,reward,done,_ = env.step(action)
    next_state = np.reshape(next_state,[1,8])
    state = next_state
    time_t += 1
    print(time_t)
    ##time.sleep(0.05)
    if done:
        break
    
print("done")

#trained_model.model.save("model.h5")
#trained_model.target_model.save("target_model.h5")    

"""
import tensorflow as tf          
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))           
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
"""
















































