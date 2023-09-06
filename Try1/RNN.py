import torch.nn.functional as F
import copy
import torch
import torch.nn as nn
"""
Input: Tensor
Output: Tensor
"""

class RNNController(nn.Module):
    def __init__(self, input_size=3, hidden_size=10, output_size=1, memory_size=10):
        super(RNNController, self).__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size

        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_data):
        _, hidden = self.rnn(input_data)
        output = self.fc(hidden.squeeze(0))
        return output
