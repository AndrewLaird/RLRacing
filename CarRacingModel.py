import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import vizdoomgym

class CarRacingModel:

    def __init__(self,num_layers):
        self.model = CarRacingTorch(num_layers)

def resize_and_grayscale(observation):
    #print("observation size",observation.shape)
    observation = observation[:210,:]
    observation = np.mean(observation,axis=2)
    observation_image = Image.fromarray(observation)
    observation_image = observation_image.resize((100,100))
    observation = np.array(observation_image)

    return observation

class CarRacingTorch(nn.Module):

    def __init__(self,num_layers):
        super(CarRacingTorch).__init__()
        self.num_layers = num_layers
        self.conv1 = nn.Conv2d(in_channels=num_layers,
                               out_channels=128 ,
                               kernel_size=5 )
        #take size
        self.pool1 = nn.AvgPool2d(kernel_size=20)
        self.pool2 = nn.AvgPool2d(kernel_size=4)


        self.dense1 = nn.Linear(in_features=128,
                                out_features=3)

    def forward(self, x):
        if(x.shape[0] == self.num_layers ):
            x = [x]
        # first reshape the image
        for i in range(len(x)):
            x[i] = resize_and_grayscale(x[i])
        x = torch.Tensor(x).reshape((-1,1, 100, 100))
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.pool2(x)
        x = x.view((-1,128))