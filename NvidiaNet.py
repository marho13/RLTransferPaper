import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, channels):
        super(Net, self).__init__()
        five = (5, 5)
        three = (3, 3)
        self.norm1 = torch.nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=24, kernel_size=five, stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=five, stride=(2,2))
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=five, stride=(2,2))
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=three)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=three)
        self.fc1 = nn.LazyLinear(100)
        self.fc2 = nn.LazyLinear(50)
        self.fc3 = nn.LazyLinear(10)

    def forward(self, x):
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = F.relu(self.fc1(torch.flatten(x, start_dim=1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net(3)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

def main():
    ############## Hyperparameters ##############
    env_name = "CarRacing-v0"
    render = False
    solved_reward = 300  # stop training if avg_reward > solved_reward
    log_interval = 20  # print avg reward in the interval
    max_episodes = 10000  # max training episodes
    max_timesteps = 1500  # max timesteps in one episode

    update_timestep = 4000  # update policy every n timesteps
    action_std = 0.5  # constant std for action distribution (Multivariate Normal)
    K_epochs = 80  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr = 0.0001  # parameters for Adam optimizer
    betas = (0.9, 0.999)

    random_seed = 42
    #############################################

    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]*env.observation_space.shape[1]*env.observation_space.shape[2]
    action_dim = env.action_space.shape[0]
    print(env.observation_space.shape[0], env.observation_space.shape[1], env.observation_space.shape[2])
    imgs = []
    imgNum = [float(x) for _ in range(10) for x in range(10)]
    labels = []
    print("Starting to load...")
    for i in imgNum:
        imgs.append([])
        labels.append([])
        for x in range(10):
            if x==int(i):
                labels[-1].append(int(i))
            else:
                labels[-1].append(0.0)
        # labels.append(int(i))
        for _ in range(env.observation_space.shape[2]):
            imgs[-1].append([])
            for __ in range(env.observation_space.shape[0]):
                imgs[-1][-1].append([])
                for ___ in range(env.observation_space.shape[1]):
                    imgs[-1][-1][-1].append(i)

    print("Done loading")
    print(len(imgs))
    for epoch in range(200):  # loop over the dataset multiple times
        print("Epoch: {}".format(epoch))
        running_loss = 0.0
        epochAcc = 0.0
        batchSize = 32
        for i in range(len(imgs)//batchSize):

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                output = net(torch.tensor(imgs[batchSize*i: batchSize*(i+1)]))
                # print(o2)
                stuff = torch.tensor(labels[batchSize*i: batchSize*(i+1)])
                loss = criterion(output, stuff)
                loss.backward()
                optimizer.step()
                stuff = torch.argmax(stuff, -1)
                output = torch.argmax(output, -1)
                batchAcc = np.sum(np.equal(output.detach().numpy(), stuff.detach().numpy()))
                epochAcc += batchAcc
                # print statistics
                running_loss += loss.item()
        print(epochAcc/len(imgs))
        print(running_loss)
    print('Finished Training')
    # print(torch.tensor(imgs[0]).dim)

    # print(torch.max(output.data, 1))

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

# output =
if __name__ == '__main__':
    this = torch.tensor([0, 1, 3, 3])
    that = torch.tensor([1, 1, 2, 3])
    sumy = np.equal(this.numpy(), that.numpy())
    print(np.sum(sumy))
    # print(np.sum(np.equal(this.data, that.data)))
    inty = 3
    intyx = [1 if x==y else 0 for x,y in zip(this.data, that.data)]
    inty += np.sum(intyx)
    print("Stuff", inty)
    # print(np.equal([0, 1, 2, 3],[1, 1, 2, 3]))
    main()