# coding : utf-8
# Author : yuxiang Zeng

import torch
from time import time


class LSTM(torch.nn.Module):
    def __init__(self, ):
        super(LSTM, self).__init__()
        self.input_dim = 5
        self.hidden_dim = 64
        self.transfer = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.lstm = torch.nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=False)
        self.fc = torch.nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, features):
        x = self.transfer(features)
        out, (hn, cn) = self.lstm(x)
        y = self.fc(out).squeeze(0)
        return y

    def predict_delay(self):
        input_data = torch.randn(32, self.input_dim)
        t1 = time()
        with torch.no_grad():  # No need to compute gradients for prediction
            self.forward(input_data)
        t2 = time()
        predicted_delay = t2 - t1
        return predicted_delay

    def get_sample(self):
        features = torch.randn(1, 1, 5)
        return features