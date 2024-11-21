# coding : utf-8
# Author : yuxiang Zeng

import math
import torch

class GCN(torch.nn.Module):
    def __init__(self, args=1):
        super(GCN, self).__init__()
        self.args = args
        self.gnn = GraphSage(5, 600, 1, 0.10, args)

    def forward(self, adjacency, features):
        output = self.gnn(adjacency, features)
        return output

    def get_sample(self):
        import numpy as np
        num_nodes = 6  # 节点数
        num_edges = 3  # 边数

        # 随机生成边的起点和终点
        src = np.random.randint(0, num_nodes, size=num_edges)  # 起点
        dst = np.random.randint(0, num_nodes, size=num_edges)  # 终点

        # 构建邻接矩阵
        adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for s, d in zip(src, dst):
            adjacency_matrix[s, d] = 1.0  # 设置边的权重，默认为1
            adjacency_matrix[d, s] = 1.0  # 如果是无向图，补充对称边

        # 生成节点特征
        adjacency_matrix = torch.as_tensor(adjacency_matrix, dtype=torch.float32).unsqueeze(0)
        features = torch.randn((1, num_nodes, 5))  # 假设每个节点有 5 维特征
        print(adjacency_matrix.shape, features.shape)
        return adjacency_matrix, features

class GraphConvolution(torch.torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, weight_init='thomas', bias_init='thomas'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = torch.torch.nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.reset_parameters()

    def reset_parameters(self):
        self.init_tensor(self.weight, self.weight_init, 'act')
        self.init_tensor(self.bias, self.bias_init, 'act')

    def forward(self, adjacency, features):
        # print(features.shape)
        support = torch.matmul(features, self.weight)
        output = torch.bmm(adjacency, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

    @staticmethod
    def init_tensor(tensor, init_type, nonlinearity):
        if tensor is None or init_type is None:
            return
        if init_type == 'thomas':
            size = tensor.size(-1)
            stdv = 1. / math.sqrt(size)
            torch.nn.init.uniform_(tensor, -stdv, stdv)
        elif init_type == 'kaiming_normal_in':
            torch.nn.init.kaiming_normal_(tensor, mode='fan_in', nonlinearity=nonlinearity)
        elif init_type == 'kaiming_normal_out':
            torch.nn.init.kaiming_normal_(tensor, mode='fan_out', nonlinearity=nonlinearity)
        elif init_type == 'kaiming_uniform_in':
            torch.nn.init.kaiming_uniform_(tensor, mode='fan_in', nonlinearity=nonlinearity)
        elif init_type == 'kaiming_uniform_out':
            torch.nn.init.kaiming_uniform_(tensor, mode='fan_out', nonlinearity=nonlinearity)
        elif init_type == 'orthogonal':
            torch.nn.init.orthogonal_(tensor, gain=torch.nn.init.calculate_gain(nonlinearity))
        else:
            raise ValueError(f'Unknown initialization type: {init_type}')


class GraphSage(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, args):
        super(GraphSage, self).__init__()
        self.args = args
        self.nfeat = input_dim
        self.nlayer = num_layers
        self.nhid = hidden_dim
        self.dropout_ratio = dropout
        weight_init = 'thomas'
        bias_init = 'thomas'

        self.gcn = torch.nn.ModuleList()
        self.norm = torch.nn.ModuleList()
        self.act = torch.nn.ModuleList()
        self.dropout = torch.nn.ModuleList()

        # 初始化第一个图卷积层
        self.gcn.append(GraphConvolution(self.nfeat, self.nhid, bias=True, weight_init=weight_init, bias_init=bias_init))
        self.norm.append(torch.nn.LayerNorm(self.nhid))
        self.act.append(torch.nn.ReLU())
        self.dropout.append(torch.nn.Dropout(self.dropout_ratio))

        # 对后续层使用相同的隐藏层维度
        for i in range(1, self.nlayer):
            self.gcn.append(GraphConvolution(self.nhid, self.nhid, bias=True, weight_init=weight_init, bias_init=bias_init))
            self.norm.append(torch.nn.LayerNorm(self.nhid))
            self.act.append(torch.nn.ReLU())
            self.dropout.append(torch.nn.Dropout(self.dropout_ratio))
        self.fc = torch.nn.Linear(self.nhid, 1)

    def forward(self, adjacency, features):
        x = features
        for i in range(0, self.nlayer):
            x = self.act[i](self.norm[i](self.gcn[i](adjacency, x)))
            x = self.dropout[i](x)
        x = x[:, 0]  # use global node
        y = self.fc(x).flatten()
        return y
