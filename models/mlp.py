# coding : utf-8
# Author : Yuxiang Zeng
import torch
from time import time

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.input_dim = 5
        self.hidden_dim = 64
        self.output_dim = 1
        self.NeuCF = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            torch.nn.LayerNorm(self.hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim // 2, self.output_dim)
        )
        self.latency = self.predict_delay()


    def forward(self, x):
        outputs = self.NeuCF(x)
        outputs = torch.sigmoid(outputs)
        return outputs

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