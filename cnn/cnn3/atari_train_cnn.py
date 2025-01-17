from cnnmodel import CNN

import numpy as np
from scipy import stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch.optim as optim


#Preprocessor(possibly inefficient)
def preprocess(states, actions):
    """Preprocess training data, saved as p"""
    states2 = []
    # Leave necessary details only
    for image in states:
        # states2.append(image[125:190, 10:150])
        states2.append(image[:, :])
    l = len(states2)

    while l % 3 != 0:
        l = l-1
    states = np.array(states2)
    acts = []
    # Stack the states in succession of 3 states pair image
    new_states = []
    for i in range(0, l, 3):
        rgb = np.dstack((states[i], states[i+1], states[i+2]))
        new_states.append(rgb)
        tmp_arr = actions[i:i+3]
        a = stats.mode(tmp_arr)
        a = a[0][0]
        acts.append(a)
    actions = np.array(acts)
    actions = torch.from_numpy(actions)
    states = np.array(new_states)

    states = torch.from_numpy(states).float()

    return states, actions

class GameData(Dataset):
    def __init__(self, name = '../data/train/RoadRunner.npz'):
        game = np.load(name)
        self.states = game['states']
        self.actions = game['actions']
        print('Starting preprocess')
        self.states, self.actions = preprocess(self.states, self.actions)
        self.n_samples = self.actions.shape[0]
        l, w, h, c = self.states.shape
        self.states = self.states.reshape([l, c, w, h])
        print('Initialization finished......')
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        return self.states[index], self.actions[index]

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 4

    trainset = GameData()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=2, drop_last=True)

    classes = tuple(np.arange(0, 10))

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    net  = CNN(3, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    print('Training starting................................')
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            if i % 10 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    print('Finished Training')

    
    PATH = './Training1.pth'
    torch.save(net.state_dict(), PATH)

    # Testing 
    #Test the model
testset = GameData(name='../data/test/RoadRunner2.npz')
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,num_workers=2)

# Test the network on the whole dataset
correct = 0
total = 0
print('Starting test.........')
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    x = 0
    for data in testloader:
        images, labels = data
        x+= 1

        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network: %d%%' % (
    100 * correct / total))