import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # the hidden is from 1 to 80 in paper
        self.fc1 = nn.Linear(in_features=input_size, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


if __name__ == '__main__':
    a = torch.rand(8, 2)
    net = Net(input_size=2)
    b = net(a)
    print(b.shape)



