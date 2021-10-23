from cnnmodel import CNN

import gym
import numpy as np
from scipy import stats
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch.optim as optim
import os
import json
from argparse import ArgumentParser
import argparse

def parse_args():
    parser = argparse.ArgumentParser("CNN experiments for Atari games")
    parser.add_argument("--seed", type=int, default=42, help="which seed to use")
    # Environment
    parser.add_argument("--env", type=str, default="PongNoFrameskip-v4", help="name of the game")
    # Core CNN parameters
    parser.add_argument("--optimizer", type=str, default="Adam", 
    					help="Name of the optimizer to be used\n Available Optimizers \n 1. Adam\n2. SGD")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam or SGD optimizer")
    parser.add_argument("--num-steps", type=int, default=int(1e6),
                        help="total number of steps to run the environment for")
    parser.add_argument("--batch-size", type=int, default=32, help="number of transitions to optimize at the same time")
    parser.add_argument("--num-epochs", type=int, default=2, help="number of epochs")
    parser.add_argument("--print-freq", type=int, default=10, help="print frequency.")
    parser.add_argument("--num-workers", type=int, default=2, help="number of workers in the network")
    parser.add_argument("--num-channels", type=int, default=4, help="number of input channels")
    parser.add_argument("--num-classes", type=int, default=6, help="number of output classes")
    return parser.parse_args()

# Save file
def path_name(env_name):
	directory = 'Models/'
	#env_name = env.unwrapped.spec.id.lower()
	env_name = env_name.lower()
	path = os.path.join(directory, env_name)
	dir_exist = os.path.isdir(path)
	dataset_name = env_name

	if not dir_exist:
		os.mkdir(path)

	count = 1
	for dirname, dirnames, filenames in os.walk(path):
		for filename in filenames:
		    print(os.path.join(dirname, filename))
		    count += 1
	#Create and save dataset
	loc = dataset_name.find('-')
	filepath = dataset_name[:loc]+str(count)
	
	return directory + env_name +'/' +  filepath


class GameData(Dataset):
	def __init__(self, name = 'pongnoframeskip1.npz'):
		##Load dataset
		game = np.load(name)
		game2 = np.load('pongnoframeskip2.npz')
		states1 = game['states'].astype(np.float32)
		states2 = game2['states'].astype(np.float32)
		print(states1.shape, states2.shape)
		self.states = np.append(states1, states2, axis = 0)
		print(self.states.shape)
		act1 = game['actions']
		act2 = game2['actions']
		self.actions = np.append(act1, act2)
		
	#        self.states = game['states'].astype(np.float32)
	#        self.actions = game['actions']
		
		self.n_samples = self.actions.shape[0]
		self.actions = torch.from_numpy(self.actions)
		self.states = torch.from_numpy(self.states)
		
	def __len__(self):
		return self.n_samples

	def __getitem__(self, index):
		return self.states[index], self.actions[index]
    

if __name__ == '__main__':
	
	args = parse_args()


	batch_size = args.batch_size
	num_workers = args.num_workers
	num_channels = args.num_channels
	num_classes = args.num_classes
	learning_rate = args.lr
	epochs = args.num_epochs
	opt_name = args.optimizer
	print_freq = args.print_freq
	
	# dataloader = DataLoader(dataset= trainset, batch_size = batch_size, shuffle = True, num_workers = num_workers)

#	trainset = GameData()
	dataset = GameData()
	
	transform = transforms.Compose(
		[transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#	train_size = int(0.8*len(dataset))
#	traindata, testdata = torch.utils.data.random_split(dataset, [train_size, test_size])

	train_size = int(0.8* len(dataset))
	test_size = len(dataset) - train_size
	trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,num_workers= num_workers)

	# Number of classes can be found using env.actions.n from gym environment
	classes = tuple(np.arange(0, num_classes))

	dataiter = iter(trainloader)
	images, labels = dataiter.next()

	net  = CNN(num_channels, num_classes) 
	criterion = nn.CrossEntropyLoss()
	if opt_name == "SGD":
		optimizer = optim.SGD(net.parameters(), lr=learning_rate)
	else:
		optimizer = optim.Adam(net.parameters(), lr=learning_rate)

	print('Training starting................................')
	for epoch in range(epochs):  # loop over the dataset multiple times
		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
		    # get the inputs; data is a list of [inputs, labels]
		    inputs, labels = data

		    # zero the parameter gradients
		    optimizer.zero_grad()

		    # forward + backward + optimize
		    outputs = net(inputs)
		    loss = criterion(outputs, labels)
		    loss.backward()
		    optimizer.step()
		    
		    # print statistics
		    running_loss += loss.item()
		    if i % print_freq == 0:    # print every print_freq mini-batches
		        print('[%d, %5d] loss: %.3f' %
		            (epoch + 1, i + 1, running_loss / print_freq))
		        running_loss = 0.0

	print('Finished Training')

	PATH = path_name(args.env)
	torch.save(net.state_dict(), PATH + '.npz')
	
	# Test Network Performance
#	testset = GameData(name='pongnoframeskip2.npz')
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,num_workers= num_workers)
	print('Starting training')
	correct = 0
	total = 0
	
	with torch.no_grad():
		for data in testloader:
			images, labels = data
			
			# calculate outputs by running images through the network
			outputs = net(images)
			# the class with the highest energy is what we choose as prediction
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	
	results = 100*(correct/total)
	print('Accuracy of the network on the test set: %d %%' % (
    	results))
    
    
	#Save environement details
	details ={
		"env": args.env,
		"optmizer": args.optimizer,
		"lr": args.lr,
		"num-steps": args.num_steps,
		"num-epochs": args.num_epochs,
		"num-workers": args.num_workers,
		"num-channels": args.num_channels,
		"num-classes": args.num_classes,
		"test-results": results
		}
	with open(PATH+'.json', 'w') as fp:
		json.dump(details, fp)
