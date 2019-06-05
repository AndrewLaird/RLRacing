import gym
from PIL import Image


if __name__ == "__main__":
    env = gym.make("CarRacing-v0")

    observation = env.reset()

    image = Image.fromarray(observation)
    image.show()

    print(env.action_space)
    for i in range(20000):
        env.render()
        action =  env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if(done):break
#stack 5 frames of the environment

class CarRacingStacked():
    def __init__(self,num_stacked):
        self.env = gym.make("CarRacing-v0")
        self.stacked = num_stacked

    def reset(self):
        observation = self.env.reset()
        return [observation for x in range(self.stacked)]


    def step(self,action):
        observations,rewards,infos = [],[],[]
        terminal = False
        for i in range(self.stacked):
            observation, reward, terminal, info = self.env.step(action)
            observations.append(observation)
            rewards.append(reward)
            if(terminal):
                break
        terminals = [terminal for x in range(self.stacked)]

        return observations,rewards,terminals,infos




