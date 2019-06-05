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