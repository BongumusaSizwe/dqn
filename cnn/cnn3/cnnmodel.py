import gym
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    """
    Implementation of a Deep Convolutional Neural Network. The architecture is as follows:
    conv1: conv2d(210, 210, kern_size = (12, 12), stride=16)
    (conv2): conv2d(210, 128, kern_size = (10, 10), stride=8)
    (conv3): conv2d(128, 64, kern_size(8, 8), stride=4)
    (conv4): conv2d(64, 64, kern_size(4, 4), stride = 2)
    (conv5): conv2d(64, 64, kern_size(3, 3), stride = 1)
    (linear1): Linear(in_features= 64*a*b, out_features = 1024, bias = True)
    (linear2): Linear(in_features= 1024, out_features = 512, bias = True)
    (linear3): Linear(in_features= 512, out_features = 256, bias = True)
    (linear4): Linear(in_features= 256, out_features = 10, bias = True)
    """
    def __init__(self, observation_space, action_space, num_layers = 5, stride_li= [8, 8, 4, 2, 1], kern_li = [8, 2, 2, 1]):
        super().__init__()
        self.conv1 = nn.Conv2d(observation_space, 210, kern_li[0], stride = stride_li[0])
        self.conv2 = nn.Conv2d(210, 128, kern_li[1], stride= stride_li[1])
        self.conv3 = nn.Conv2d(128, 64, kern_li[2], stride = stride_li[2])
        self.conv4 = nn.Conv2d(64, 64, kern_li[3], stride = stride_li[3])
        self.conv5 = nn.Conv2d(64, 64, kern_li[3], stride = stride_li[4])
        def conv_returns(w, k, s, p = 0):
            '''
            Returns the shape of the output after convolutin
            
            w: Input size
            k: kernel size
            s: stride
            p: pad sie
            '''
            return int((w-k+2*p)/s + 1 )
        # Input is an image of 210 by 160
        self.a =   210
        self.b = 160
        for i in range(num_layers - 1):
            self.a = conv_returns(self.a, kern_li[i], stride_li[i])
            self.b = conv_returns(self.b, kern_li[i], stride_li[i])
        # print(self.a, self.b)
        self.linear1 = nn.Linear(64 * self.a *self.b, 1024, bias=True)
        self.linear2 = nn.Linear(1024, 512, bias=True)
        self.linear3 = nn.Linear(512, 64, bias=True)
        self.linear4 = nn.Linear(64, action_space, bias=True) #10 is the action space

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * self.a * self.b)  # flatten
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        return x
