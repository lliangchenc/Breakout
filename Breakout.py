import gym
import time
import cv2
import numpy as np
from DQN_model import DQN

NUM_ACTIONS = 4

env = gym.make('Breakout-v0')
env.reset()
print env.action_space, env.observation_space
done = False
reward = 0.0
total_reward = 0.0

#state, reward, done, info = env.step(2)

def rgb2grey(img):
    #print "RGB", np.shape(np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) / 255.0)
    return np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

myDQN = DQN(NUM_ACTIONS)
state, reward, done, info = env.step(env.action_space.sample())
myDQN.initState(rgb2grey(state))
while True:
    action = myDQN.getAction()
    state, reward, done, info = env.step(action)
    temp_action = np.zeros(NUM_ACTIONS)
    temp_action[action] += 1.0
    total_reward += reward
    if done:
        reward -= 10000
        myDQN.feedFrame(rgb2grey(state), temp_action, reward, done)
    else:
        myDQN.feedFrame(rgb2grey(state), temp_action, total_reward, done)
    env.render()
    #time.sleep(0.05)
    if done :
        print "REWARD : ", total_reward
        total_reward = 0.0
        env.reset()
