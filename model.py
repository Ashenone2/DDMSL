import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb
import scipy.sparse as sp


class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """

    def __init__(
            self, optimizer, patience=10, min_lr=5e-8, factor=0.5
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True
        )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)



from torch.nn import BatchNorm1d, Dropout, ReLU, Linear, Conv1d, ConvTranspose1d, LayerNorm




class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.W = nn.Parameter(torch.randn(input_dim, output_dim))  # 权重参数
        self.b = nn.Parameter(torch.randn(output_dim))  # 偏差参数

    def forward(self, adj, X):
        # 计算对称归一化的邻接矩阵和度矩阵
        Relu = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
        deg = torch.sum(adj, dim=1, keepdim=True)
        norm_adj = adj / torch.sqrt(deg) / torch.sqrt(deg.transpose(1, 0))
        # 执行图卷积操作
        X = X.matmul(self.W)

        # norm_adj = torch.unsqueeze(norm_adj, dim=0)  # 扩展为 (1, 100, 100)
        # norm_adj = norm_adj.repeat([X.shape[0], 1, 1])  # 复制成 2 份
        #
        # out = torch.bmm(norm_adj, X)
        out = []
        for i, x in enumerate(X):
            out.append(norm_adj @ x)
        out = torch.stack(out, dim=0)
        out = out + self.b  # 添加偏差项
        out = Relu(out)  # 使用ReLU激活函数
        return out

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.W = nn.Parameter(torch.randn(input_dim, output_dim))  # 权重参数
        self.b = nn.Parameter(torch.randn(output_dim))  # 偏差参数

    def forward(self,adj, X):
        # 计算对称归一化的邻接矩阵和度矩阵
        Relu = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
        deg=torch.sum(adj, dim=1, keepdim=True)
        norm_adj = adj / torch.sqrt(deg) / torch.sqrt(deg.transpose(1, 0))
        X=X.matmul(self.W)
        out = []
        for i, x in enumerate(X):
            out.append(norm_adj @ x)
        out = torch.stack(out, dim=0)+X
        out = out + self.b  # 添加偏差项
        out = Relu(out)  # 使用ReLU激活函数
        return out


def creat_sin_cos_emb(dim):
    assert dim % 2 == 0, "wrong!"
    n_pos_vec = torch.arange(21, dtype=torch.float)
    position_embedding = torch.zeros(n_pos_vec.shape[0], dim, dtype=torch.float)
    omega = torch.arange(dim // 2, dtype=torch.float)
    omega /= dim / 2.
    omega = 1. / (10000 ** omega)
    out = n_pos_vec[:, None] @ omega[None, :]
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    position_embedding[:, 0::2] = emb_sin
    position_embedding[:, 1::2] = emb_cos
    return position_embedding


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GCNB(torch.nn.Module):

    def __init__(self, g):
        super(GCNB, self).__init__()
        self.g = g
        self.linears = nn.Sequential(torch.nn.utils.spectral_norm(nn.Linear(3, 128)),
                                     nn.LeakyReLU(),
                                     Dropout(0.5), )
        self.linears2 = nn.Sequential(
            torch.nn.utils.spectral_norm(Linear(128, 32)),
        )
        self.linears3 = nn.Sequential(
            torch.nn.utils.spectral_norm(Linear(32, 1)),
        )

        self.conv1 = torch.nn.utils.parametrizations.spectral_norm(GraphConvolution(128, 128), name='W')
        self.conv2 = torch.nn.utils.parametrizations.spectral_norm(GraphConvolution(128, 128), name='W')
        self.conv3 = torch.nn.utils.parametrizations.spectral_norm(GraphConvolution(128, 128), name='W')
        self.batch_norm = BatchNorm1d(g.shape[0], track_running_stats=False)

    def forward(self, g, data, t):
        # Relu = torch.nn.LeakyReLU(negative_slope=0.08, inplace=False)
        t_emb = creat_sin_cos_emb(128).to(device)
        Relu = torch.nn.Mish(inplace=False)
        x = data.to(torch.float32)

        self.g = g
        # print('shape0', x.shape)
        x = self.linears(x)

        ebd = t_emb[t]
        ebd = ebd.repeat([x.shape[0], g.shape[0], 1])
        x = x + ebd
        x1 = self.conv1(g, x)
        x11 = self.batch_norm(x1)

        x2 = Relu(x11) + x
        Dropout(0.5)
        xt2 = self.conv2(g, x2)
        x22 = self.batch_norm(xt2)
        x_21 = Relu(x22) + x2
        Dropout(0.5)

        xt2 = self.conv2(g, x_21)
        x22 = self.batch_norm(xt2)
        x_22 = Relu(x22) + x_21
        Dropout(0.5)

        xt2 = self.conv2(g, x_22)
        x22 = self.batch_norm(xt2)
        x_23 = Relu(x22) + x_22
        Dropout(0.5)

        xt2 = self.conv2(g, x_23)
        x22 = self.batch_norm(xt2)
        x_24 = Relu(x22) + x_23
        Dropout(0.5)
        x3 = x_24

        x3 = self.conv3(g, x3)
        x33 = self.batch_norm(x3)
        # print('shape4', x3.shape)
        # 线性层2

        x4 = Relu(x33) + x_24
        Dropout(0.5)
        x4 = self.linears2(x4)
        Dropout(0.5)
        x4 = torch.relu(x4)
        # 线性层3
        x4 = self.linears3(x4)
        Dropout(0.5)
        x4 = torch.sigmoid(x4)
        x4 = torch.squeeze((x4))
        # print('4', x4.shape)
        return x4






