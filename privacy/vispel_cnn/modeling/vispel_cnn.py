import torch
import torch.nn as nn
import torch.nn.functional as F

class VISPEL(nn.Module):

    def __init__(self):

        super(VISPEL, self).__init__()

        self.conv1 = nn.Conv2d(4, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32*4*4, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x

def pearson_loss(output, target):

    loss = 0
    for k in range(output.size()[1]):
        x = output[k,:]
        y = target[k,:]

        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        loss += torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

    loss = -torch.log(torch.abs(loss/output.size()[0]))

    return loss