# coding : utf-8
# Author : yuxiang Zeng

import torch
from time import time

class BIRNN(torch.nn.Module):
    def __init__(self, ):
        super(BIRNN, self).__init__()
        self.input_dim = 5
        self.hidden_dim = 64
        self.transfer = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.lstm = torch.nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(self.hidden_dim * 2, 1)
        self.latency = self.predict_delay()

    def forward(self, dnn_seq):
        x = self.transfer(dnn_seq)
        out, (hn, cn) = self.lstm(x)
        hn_fwd = hn[-2, :, :]  # 前向的最后隐藏状态
        hn_bwd = hn[-1, :, :]  # 后向的最后隐藏状态
        hn_combined = torch.cat((hn_fwd, hn_bwd), dim=1)  # 形状: (batch_size, hidden_dim * 2)
        y = self.fc(hn_combined)  # 形状: (batch_size, output_dim)
        return y

    def predict_delay(self):
        input_data = torch.randn(1, 1, self.input_dim)  # Adjust shape for LSTM input
        t1 = time()
        with torch.no_grad():  # No need to compute gradients for prediction
            self.forward(input_data)
        t2 = time()
        predicted_delay = t2 - t1
        return predicted_delay


    def get_sample(self):
        features = torch.randn(1, 1, 5)
        return features
