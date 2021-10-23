import gym
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    """
    Implementation of a Deep Convolutional Neural Network. The architecture is the same as that used in the
    Nature DQN paper
	(channels): Number of channels per input array
	(action_space): Expected number of output categories
	(width): Image width
	(height): Image height
	(num_layers): Number of layers in the network
	(stride_li): A list containing strides for each layer
	(kern_li):  A list containing kernel sizes for each layer
	================================================================
    (conv1): Conv2d(210, 32, kernel_size=(8, 8), stride=(4, 4))
    (conv2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
    (conv3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
    (linear1): Linear(in_features=3136, out_features=512, bias=True)
    (linear2): Linear(in_features=512, out_features=6, bias=True)

    """
    def __init__(self, channels, action_space, width = 84, height = 84, num_layers = 3, stride_li= [8, 2, 2], kern_li = [8, 3, 3]):

        super().__init__()
        self.conv1 = nn.Conv2d(channels, 32, kern_li[0], stride = stride_li[0])
        self.conv2 = nn.Conv2d(32, 64, kern_li[1], stride= stride_li[1])
        self.conv3 = nn.Conv2d(64, 64, kern_li[2], stride = stride_li[2])
        def conv_returns(w, k, s, p = 0):
            '''
            Returns the shape of the output after convolutin
            
            w: Input size
            k: kernel size
            s: stride
            p: pad sie
            '''
            return int((w-k+2*p)/s + 1 )
        self.a =   width
        self.b = height
        for i in range(num_layers - 1):
            self.a = conv_returns(self.a, kern_li[i], stride_li[i])
            self.b = conv_returns(self.b, kern_li[i], stride_li[i])
        # print(self.a, self.b)
        self.linear1 = nn.Linear(64 * self.a *self.b, 512)
        self.linear2 = nn.Linear(512, action_space) #10 is the action space

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64 * self.a * self.b)  # flatten
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

