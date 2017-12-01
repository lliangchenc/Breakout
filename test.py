import gym
import time

env = gym.make('Breakout-v0')
env.reset()
print env.action_space, env.observation_space
done = False
reward = 0.0
total_reward = 0.0
env.step(3)
#state, reward, done, info = env.step(2)

for _ in range(100000):
    state, reward, done, info = env.step(env.action_space.sample())
    total_reward += reward
    env.render()
    #time.sleep(0.05)
    if done :
        print "REWARD : ", total_reward
        total_reward = 0.0
        env.reset()


