import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)   # input layer > hidden layer
        self.fc2 = nn.Linear(128, 64)      # hidden > hidden
        self.fc3 = nn.Linear(64, 10)       # hidden > output (10 digits)

    def forward(self, x):
        x = x.view(-1, 28*28)  # flatten 28x28 > 784
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x # logits for each class