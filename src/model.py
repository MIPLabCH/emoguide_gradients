"""
Copyright © 2023 Chun Hei Michael Chan, MIPLab EPFL
"""


import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


class MLP(nn.Module):
    """
    General MultiLayerPerceptron ~10k parameters
    """

    def __init__(self, idim):
        super().__init__()
        self.fc1 = nn.Linear(idim, 120)
        self.fc2 = nn.Linear(120,32)
        self.fc3 = nn.Linear(32,1)

    def forward(self, x):
        """
        Feedforward of NN
        """
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        x = x.squeeze(1)

        return x

class CombinePearsonLoss(nn.Module):
    """
    Custom loss with weighted combination of MSE and inverse Pearson correlation
    """

    def __init__(self, weight=None, size_average=True):
        super(CombinePearsonLoss, self).__init__()

    def forward(self, inputs, targets):
        """
        Loss evaluation
        """
        num = torch.sum((inputs - inputs.mean()) * (targets-targets.mean()))
        den = torch.sqrt(torch.sum((inputs - inputs.mean())** 2) * torch.sum((targets - targets.mean())** 2))
        #returning the results
        total = 10*(1 - num/den) + nn.MSELoss()(inputs,targets)
        
        return total