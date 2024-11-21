# coding : utf-8
# Author : yuxiang Zeng

from time import time
import torch

class GRU(torch.nn.Module):
    def __init__(self, ):
        super(GRU, self).__init__()
        self.input_dim = 5
        self.hidden_dim = 64
        self.transfer = torch.nn.Linear(5, self.hidden_dim)
        self.lstm = torch.nn.GRU(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.fc = torch.nn.Linear(self.hidden_dim, 1)

    def forward(self, features):
        x = self.transfer(features)
        out, op_embeds = self.lstm(x)
        y = self.fc(out)
        return y

    def predict_delay(self):
        input_data = torch.randn(32, 5)
        t1 = time()
        with torch.no_grad():  # No need to compute gradients for prediction
            self.forward(input_data)
        t2 = time()
        predicted_delay = t2 - t1
        return predicted_delay

    def get_sample(self):
        features = torch.randn(1, 1, 5)
        return features