import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch



class LS_D(nn.Module):
    def __init__(self):
        super(LS_D, self).__init__()
        self.fc1 = nn.Linear(142, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
