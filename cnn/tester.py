'''This file runs uses different arguments to test a cnn model'''

import atari_train_cnn

#Number of classes for pong is 4

learning_rates = ["1e-1", "1e-2", "1e-3", "1e-4", "1e-5", "1e-6"]

batch_sizes = ["4", "8", "16", "32", "64", "128"]

num_epochs = ["2", "4", "8", "16", "32", "64", "128"]

optimizers = ["Adam", "SGD"]
num_workers = ["2", "4", "8"]

# num_layers = ["4"]

# --num-classes 6 --num-epochs 10 --print-freq 20
#Running with different learning rates

print("Testing learning rates")

for l in learning_rates:
#    print("lr = " + l)
    atari_train_cnn.main()
 #   atari_train_cnn.main(["--num-classes", "6", "--num-epochs", "16", "--lr", l, "--print-freq", "20"])
print("Done")

