import ctypes

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import copy
import time
import networkx as nx
import random
from scipy.special import softmax
from scipy.sparse import csr_matrix
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from sklearn import model_selection

from main.utils import load_dataset, InverseProblemDataset, adj_process
from main.model.model import GNNModel, VAEModel, DiffusionPropagate, Encoder, Decoder
from main.inversemodel import InverseModel, ForwardModel,MLP,InverseModel0,GCN,GCNB,LRScheduler,GCN_con,GCNB_1
from main.inference import model_train, inference
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import warnings
import platform
from ceshiPT import NEDMP_pred
warnings.filterwarnings("ignore")


def creat_Pt(iterations, beta, gamma, m, g,  infected_nodes,n):
    MAX_NODES = 20000
    if platform.system().lower() == 'windows':
        mydll = ctypes.CDLL('./Dll1.dll', winmode=0)
    # 定义函数参数类型
    mydll.my_dll_add.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int * MAX_NODES)
    ]
    mydll.my_dll_add.restype = None
    adj = nx.adjacency_matrix(g).todense()
    adj_matrix = np.array(adj, dtype=np.int32)
    num_nodes = adj.shape[0]
    adj_matrix_c = (ctypes.POINTER(ctypes.c_int) * MAX_NODES)()
    for i in range(num_nodes):
        adj_matrix_c[i] = adj_matrix[i].ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    # 将infected_nodes_c创建为一个ctypes数组
    infected_nodes_c = (ctypes.c_int * MAX_NODES)(*infected_nodes)
    num_infected_nodes=len(infected_nodes)
    # infected_nodes_c = (ctypes.c_int * MAX_NODES)(*map(ctypes.c_int, infected_nodes))
    # 调用函数
    "调用函数，输入参数依次为：每次感染迭代时长、感染率、恢复率、邻接矩阵、节点数、初始感染节点（List）、初始感染节点数，并行线程数"
    mydll.my_dll_add(iterations, beta, gamma, m, adj_matrix_c, num_nodes, infected_nodes_c,num_infected_nodes,n)
    with open("./output.txt", "r") as f:
        lines = f.readlines()

    result = []
    for line in lines:
        node_values = line.split(':')[1].split()  # 获取节点数值
        node_values = [float(x) for x in node_values]  # 将数值从 str 转为 float
        node_values = [node_values[i:i+3] for i in range(0, len(node_values), 3)]
        result.append(node_values)

    lst=result
    lst_len = len(lst)
    sub_lst_len = len(lst[0])
    result = []
    for i in range(sub_lst_len):
        sub_lst = []
        for j in range(lst_len):
            sub_lst.append(lst[j][i])
        result.append(sub_lst)
    return result

def creat_mat(list):
    "输入一个列表的矩阵，输出有这些矩阵沿对角线组成的大矩阵"
    p = list[0]
    q = list[1]
    result = np.block([[p, np.zeros_like(p)], [np.zeros_like(q), q]])
    for i in range(2,len(list)):
        result = np.block([[result, np.mat(np.zeros((len(result), 3)))], [np.mat(np.zeros((3, len(result)))), list[i]]])
    return result
def build_karate_club_graph():
    g = nx.Graph()
    # add 34 nodes into the graph; nodes are labeled from 0~33

    # all 78 edges as a list of tuples
    edge_list = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
                 (4, 0), (5, 0), (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
                 (7, 2), (7, 3), (8, 0), (8, 2), (9, 2), (10, 0), (10, 4),
                 (10, 5), (11, 0), (12, 0), (12, 3), (13, 0), (13, 1), (13, 2),
                 (13, 3), (16, 5), (16, 6), (17, 0), (17, 1), (19, 0), (19, 1),
                 (21, 0), (21, 1), (25, 23), (25, 24), (27, 2), (27, 23),
                 (27, 24), (28, 2), (29, 23), (29, 26), (30, 1), (30, 8),
                 (31, 0), (31, 24), (31, 25), (31, 28), (32, 2), (32, 8),
                 (32, 14), (32, 15), (32, 18), (32, 20), (32, 22), (32, 23),
                 (32, 29), (32, 30), (32, 31), (33, 8), (33, 9), (33, 13),
                 (33, 14), (33, 15), (33, 18), (33, 19), (33, 20), (33, 22),
                 (33, 23), (33, 26), (33, 27), (33, 28), (33, 29), (33, 30),
                 (33, 31), (33, 32)]
    # add edges two lists of nodes: src and dst
    src, dst = tuple(zip(*edge_list))
    for i in range(len(src)):
        g.add_edge(dst[i], src[i])
    return g
def construct_graph(graph_filepath):
    graph = nx.Graph()
    graph_edges = []
    with open(graph_filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip().split(",")
            line=list(map(int, line))
            line=[i-1 for i in line]
            graph_edges.append(tuple(line))
            for i in line:
                if graph.has_node(i):
                    continue
                else:
                    graph.add_node(i)
    graph.add_edges_from(graph_edges)
    return graph
def reg_loss(model,lamda=0):
    reg_loss=0.0
    for param in model.parameters():
        reg_loss=reg_loss+torch.sum(param**2)
    return lamda*reg_loss
def pt2Qt(pt):
    Qt=[]
    l=len(pt)
    for t in range(l-1):
        tempt=[]
        for j in range(len(pt[t])):
            pi=1-pt[t+1][j][0]/(pt[t][j][0]+0.000000000000000001)
            pr=(pt[t+1][j][2]-pt[t][j][2])/(pt[t][j][1]+0.000000000000000001)
            Qti=np.mat([[1-pi,pi,0],[0,1-pr,pr],[0,0,1]])
            Qti=Qti.tolist()
            tempt.append(Qti)
        Qt.append(tempt)
    return Qt

# def genraQ(g,seedlist):
#     infect_rate,recover_rate,t=0.03,0.015,21
#     Qt=[]
#     qt=[]
#     prob = NEDMP_pred(g, seedlist)
#     print(seedlist[0])
#     print(prob[0][20])
#     for p in prob:
#         q=pt2Qt(p)
#         qt.append(q)
#     return qt
def genraQ(g,seedlist,t):
    infect_rate,recover_rate,t=0.03,0.015,t+1
    Qt=[]
    qt=[]
    for i,seed in enumerate(seedlist):
        prob=creat_Pt(t, infect_rate,recover_rate, 10000, g, seed,12)
        q=pt2Qt(prob)
        qt.append(q)
    return qt


print('Is GPU available? {}\n'.format(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def npz2list(path):
    list1 = np.load(path,allow_pickle=True)
    arr_0 = list1['arr_0']
    return arr_0

test_set=npz2list(r'F:/diffusion_source_detect/diff_dataset/karate/test_set.npz')
test_set=torch.Tensor(test_set)
# test_set=test_set.permute(0,1,3,2)
n=50

# strong=torch.zeros(test_set.shape[0],test_set.shape[1]*n,test_set.shape[2],test_set.shape[3])
# strongx=[]
# strongy=[]
# from copy import deepcopy
# for i,set in enumerate(test_set):
#     for j,data in enumerate(set):      #2X198X16
#         x=data[:,0]
#         y=data[:,15]
#         for c in range(n):
#             strongx.append(x)
#             outy = deepcopy(y)
#             for k in range(len(y)):
#
#                 if y[k]==1:
#                     if random.uniform(0, 1)<0.05:
#                         outy[k]=0
#                 if y[k]==2:
#                     if random.uniform(0, 1)<0.025:
#                         outy[k]=1
#
#             strongy.append(outy)
# strongx=torch.stack(strongx,dim=0)
# strongy=torch.stack(strongy,dim=0)
# strong[0][:,:,0]=strongx
# strong[0][:,:,15]=strongy
# test_set=strong
# print(trainset[0][0])


parser = argparse.ArgumentParser(description="SLVAE")
datasets = ['jazz_SIR', 'jazz_SI', 'cora_ml_SIR', 'cora_ml_SI', 'power_grid_SIR', 'power_grid_SI',
            'karate_SIR', 'karate_SI', 'jazz_SI', 'netscience_SI']
parser.add_argument("-d", "--dataset", default="karate_SIR", type=str,
                    help="one of: {}".format(", ".join(sorted(datasets))))

args = parser.parse_args(args=[])

import pickle

with open('data/' + args.dataset + '.SG', 'rb') as f:
    graph = pickle.load(f)
adj, _, prob_matrix = graph['adj'].toarray(), graph['inverse_pairs'], graph['prob'].toarray()






#
# train_set, rest_set,Qtrain_set,Qrest_set=model_selection.train_test_split(inverse_pairs,Qt,train_size=0.8,test_size= 0.2,shuffle=True,random_state=0)
# test_set,val_set,Qtest_set,Qval_set=model_selection.train_test_split(rest_set,Qrest_set,train_size=0.1,test_size= 0.9,shuffle=True,random_state=0)

# train_set=inverse_pairs
# test_set=inverse_pairs
# val_set=test_set
# Qtrain_set=Qt

vae_model = MLP()

gnn_model = GNNModel(input_dim=5,
                     hiddenunits=[64, 64],
                     num_classes=1,
                     prob_matrix=prob_matrix)


propagate = DiffusionPropagate(prob_matrix, niter=2)

from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def loss_all(x, x_hat, log_var, mean, y_hat, y):
    # forward_loss = F.mse_loss(y_hat, y, reduction='sum')
    # monotone_loss = torch.sum(torch.relu(y_hat-y_hat[0]))
    reproduction_loss = F.binary_cross_entropy(x_hat, x, reduction='mean')
    #KLD = -0.5*torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    total_loss = reproduction_loss
    return reproduction_loss,  total_loss
"***********************************"

def xt_X_QTrans(xt, Qt):
    "xt乘以Q的转置"
    "q(x_t|x_t-1)=xt@Q_t^T"
    xt=torch.tensor(xt,dtype=torch.float)
    xt=xt.to(device)
    xt = torch.unsqueeze(xt, dim=2)
    Qt=torch.tensor(Qt).to(device)
    QtT=torch.zeros_like(Qt)
    for i in range(len(Qt)):
        for j in range(len(Qt[i])):
            QtT[i][j]=torch.transpose(torch.Tensor(Qt[i][j]), 0, 1)
        QtT[i]=torch.Tensor(QtT[i])
    QtT=torch.tensor(QtT,dtype=torch.float)
    QtT=QtT.to(device)
    result=torch.zeros_like(xt)
    for i,x in enumerate(xt):
        result[i]=torch.bmm(x,QtT[i])
    result=torch.squeeze(result,2)
    return result

def xt_X_Q(xt, Qt):
    "xt乘以Q"
    xt = torch.unsqueeze(xt, dim=2)
    xt=xt.to(torch.float)
    Qt = torch.tensor(Qt,dtype=torch.float)
    result=torch.zeros_like(xt)
    for i,x in enumerate(xt):
        result[i]=torch.bmm(x,Qt[i])
    result=torch.squeeze(result,2)
    return result

def Q_mbmm(Q1,Q2):####Q:51X34X3X3
    "Q1 Q2并行乘法"
    Q1=torch.tensor(Q1,dtype=torch.float)
    Q2 = torch.tensor(Q2, dtype=torch.float)
    Q=torch.zeros_like(Q2)
    for i,q in enumerate(Q1):
        Q[i]=torch.bmm(q,Q2[i])
    return Q

def Qtba(Q,t):

    "Q:0,1,2,3对应的实际上是1,2,3,4"
    "Qt_ba连乘"
    if t==1:
        result = Q_mbmm(Q[:, 0, :, :, :, ], Q[:,1, :, :, :, ])  ##Q1xQ2
        return result
    if t==2:
        result = Q_mbmm(Q[:, 0, :, :, :, ], Q[:,1, :, :, :, ])  ##Q1xQ2
        result=Q_mbmm(result, Q[:,2, :, :, :, ])
        return result
    elif t>1 and t!=2:
        result = Q_mbmm(Q[:, 0, :, :, :, ], Q[:, 1, :, :, :, ])
        for i in range(2,t):
            #### Qt 51X34X3X3
            result=Q_mbmm(result,Q[:,i, :, :, :, ])
        return result
    else:
        result=Q[:,t,:,:,:,]
        return result

def nnout(x):
    "计算P_theta(x_0^(j)|x_t)"
    xi=x
    xs=torch.ones_like(xi)
    xs=xs-xi
    return xs,xi
def tensormul(a,n):
    "把一个张量复制n次"
    result=[]
    while True:
        result.append(a)
        if len(result)>=n:
            break
    return torch.stack(result)
def transform_tensor(tensor):
    tensor=tensor.tolist()
    result=[]
    for tens in tensor:
        result.append([[1-x, x, 0] for x in tens])
    return torch.tensor(result,dtype=torch.float)
def normalize_tensor(tensor):
    tensor = tensor.tolist()
    result = []
    for tens in tensor:
        for j,x in enumerate(tens):
            if sum(x)==0:
                tens[j]=[1,0,0]
        result.append([[x/(sum(row)) for x in row] for row in tens])
    return torch.tensor(result,dtype=torch.float)
def nntheta(xt,Q,x_pred,t):
    "计算P_theta(x_t-1|x_t)"
    q_xt_xt_1=xt_X_QTrans(xt, Q[:,t-1,:,:,:,])    #50x34x3
    Q_ba_t_1=Qtba(Q,t-2)    #50x34x3x3
    xs,xi=nnout(x_pred)     #50x34
    ps=torch.tensor([1,0,0])
    x0_s=tensormul(ps,xs.shape[1]).to(device)
    x0_s=tensormul(x0_s,xs.shape[0]).to(device) #50X34x3
    pi=torch.tensor([0,1,0]).to(device)
    x0_i=tensormul(pi,xi.shape[1]).to(device)
    x0_i = tensormul(x0_i, xi.shape[0]).to(device)
    "计算x_0^j@Q_t_1_ba*ptheta^j"
    q_ps=xt_X_Q(x0_s, Q_ba_t_1) .to(device)   #50x34x3
    q_pi=xt_X_Q(x0_i, Q_ba_t_1).to(device)

    xs,xi=torch.unsqueeze(xs,2),torch.unsqueeze(xi,2)
    rs,ri=torch.zeros_like(q_ps),torch.zeros_like(q_ps)
    for i,q in enumerate(q_ps):
        a = q * xs[i]
        rs[i]=a
    for i,q in enumerate(q_pi):
        ri[i]=q*xi[i]
    ptheta=ri+rs#50x34x3
    ptheta=ptheta.to(device)
    "q_xt_xt_1@ptheta"
    hamada=q_xt_xt_1*ptheta.to(device)

    p_pri=normalize_tensor(hamada).to(device)
    return p_pri   #50x34x3


def q_prior(xt,Q,x0,t):
    x0=torch.nn.functional.one_hot(x0.to(torch.int64), num_classes=3).to(device)
    distru1=xt_X_QTrans(xt, Q[:,t-1,:,:,:,]).to(device)
    Qba_t_1=Qtba(Q,t-2)
    Qba_t_1=torch.tensor(Qba_t_1).to(device)
    distru2=xt_X_Q(x0, Qba_t_1).to(device)
    hamada=distru1*distru2
    q_pri = normalize_tensor(hamada).to(device)

    return q_pri

def PandQ(x,y, Q, x_pred, t):
    x = torch.nn.functional.one_hot(x.to(torch.int64), num_classes=3).to(device)
    distru1 = xt_X_QTrans(y, Q[:, t - 1, :, :, :, ]).to(device)
    Qba_t_1 = Qtba(Q, t - 2)
    Qba_t_1 = torch.tensor(Qba_t_1).to(device)
    distru2 = xt_X_Q(x, Qba_t_1).to(device)
    hamada = distru1 * distru2  # 50x34x3
    q_pri = normalize_tensor(hamada).to(device)


    "计算P_theta(x_t-1|x_t)"
    q_xt_xt_1=distru1   #50x34x3
    xs,xi=nnout(x_pred)     #50x34
    ps=torch.tensor([1,0,0])
    x0_s=tensormul(ps,xs.shape[1]).to(device)
    x0_s=tensormul(x0_s,xs.shape[0]).to(device) #50X34x3
    pi=torch.tensor([0,1,0]).to(device)
    x0_i=tensormul(pi,xi.shape[1]).to(device)
    x0_i = tensormul(x0_i, xi.shape[0]).to(device)
    "计算x_0^j@Q_t_1_ba*ptheta^j"
    q_ps=xt_X_Q(x0_s, Qba_t_1) .to(device)   #50x34x3
    q_pi=xt_X_Q(x0_i, Qba_t_1).to(device)

    xs,xi=torch.unsqueeze(xs,2),torch.unsqueeze(xi,2)
    rs,ri=torch.zeros_like(q_ps),torch.zeros_like(q_ps)
    for i,q in enumerate(q_ps):
        a = q * xs[i]
        rs[i]=a
    for i,q in enumerate(q_pi):
        ri[i]=q*xi[i]
    ptheta=ri+rs#50x34x3
    ptheta=ptheta.to(device)
    "q_xt_xt_1@ptheta"
    phamada=q_xt_xt_1*ptheta.to(device)
    p_pri=normalize_tensor(phamada).to(device)
    return q_pri,p_pri
def Lvb_loss(x,y, Q, x_pred, t):
    if t>0:
        # "P_theta(x_t_1|x_t)"
        # p0 = nntheta(y, Q, x_pred, t)
        # "q(x_t_1|x_t,x_0)"
        # pr = q_prior(y, Q, x, t)
        pr,p0=PandQ(x,y, Q, x_pred, t)
    return p0
import heapq
def top_k_indices(tensor, k=3):
    """
    输出张量中前k大的索引
    """

    idx=heapq.nlargest(k, range(len(tensor)), tensor.__getitem__)
    return idx


adj, _, prob_matrix = graph['adj'].toarray(), graph['inverse_pairs'], graph['prob'].toarray()
def cor():
    filename=r"F:\diffusion_source_detect\diff_dataset\twin\out.txt"
    g = nx.Graph()
    with open(filename, 'r') as f:
        node1 = []
        node2 = []
        for line in f:
            nodes = line.strip().split()
            if len(nodes) == 2:
                node1.append(nodes[1])
                node2.append(nodes[0])
        for i in range(len(node1)):
            g.add_edge(node1[i], node2[i])

    id_map = {old_id: new_id for new_id, old_id in enumerate(g.nodes())}
    g = nx.relabel_nodes(g, id_map)
    return g


adj=npz2list('./dataset/karate/adj.npz')
g= nx.from_numpy_array(adj)
# g = cor()



import dgl
g0=torch.tensor(nx.adjacency_matrix(g).todense(),dtype=torch.float).to(device)
g1=dgl.from_networkx(g).to(device)
print(g)
model=GCNB_1(g0).to(device)
PATH="./model/SIRnet.ckpt"
model.load_state_dict(torch.load(PATH))

print(model)
MSE=torch.nn.MSELoss(reduction='mean')
test_sample_number = test_set[:].shape[0] * test_set[:].shape[1]

model.eval()
val_precision_all = 0
val_recall_all = 0
val_f1=0
val_auc=0
mark=0
mse=0


re_overall = 0
kld_overall = 0
total_overall = 0
precision_all = 0
recall_all = 0
f1=0
roc_auc=0
test_precision_all = 0
test_recall_all = 0
precision_t = 0
recall_t = 0
f1_t=0
roc_auc_t=0
test_precision_t = 0
test_recall_t = 0
process=[]
print(len(test_set))
print(test_set.shape)
threshhold=0.1
for test_batch_idx, test_data_pair in tqdm(enumerate(test_set)):
    model.eval()
    xt_1=[]
    gt=[]
    kkk=0
    with torch.no_grad():
        for t in  tqdm(range(20,0,-1)):
            gtru= test_data_pair[:, :, t].to(device)
            gt.append(gtru[kkk].cpu().detach())
            test_x = test_data_pair[:, :, 0].float().to(device)  # x[0]是所有x[i]的超集
            if t==1:
                gt.append(test_x[kkk].cpu().detach())
            if t==20:
                test_y = test_data_pair[:, :, t].to(device)
                test_y0=test_y
                test_y = torch.nn.functional.one_hot(test_y.to(torch.int64), num_classes=3)

                xt_1.append(test_y[kkk].cpu().detach())
                test_x_true = test_x.cpu().detach()
                test_x_hat = model(g1,test_y0,t).to(device)
                test_x_pred = test_x_hat.cpu().detach()
                seedlist = []
                for i in range(test_x_true.shape[0]):
                    # seed = top_k_indices(test_x_pred[i], k=int(test_x_true.shape[1] * 0.1))
                    # seedlist.append(seed)
                    test_x_pred[i][test_x_pred[i] > threshhold] = 1
                    test_x_pred[i][test_x_pred[i] != 1] = 0

                seedlist=[]
                for i in range(test_x_true.shape[0]):
                    seed=np.where(test_x_pred[i]==1)
                    seed=np.array(seed)
                    seed=seed.tolist()
                    seed=seed[0]
                    seedlist.append(seed)
                start_time = time.time()
                Q=genraQ(g, seedlist,t)
                end_time = time.time()  # 记录结束时间
                # print("程序运行时间：%.6f秒" % (end_time - start_time))  # 打印运行时间，保留小数点后六位
                Q=torch.tensor(Q).to(device)
                p0=nntheta(test_y,Q,test_x_hat,t)
                p0=torch.nn.functional.gumbel_softmax(torch.log(p0),0.001,True)
                test_y=p0
                xt_1.append(test_y[kkk].cpu().detach())

            else:
                test_x_true = test_x.cpu().detach()

                test_x_hat = model(g1, test_y0 , t).to(device)
                test_x_pred = test_x_hat.cpu().detach()
                seedlist = []
                # for i in range(test_x_true.shape[0]):
                #     seed=top_k_indices(test_x_pred[i], k=int(test_x_true.shape[1]*0.1))
                #     seedlist.append(seed)
                for i in range(test_x_true.shape[0]):
                    test_x_pred[i][test_x_pred[i] > threshhold] = 1
                    test_x_pred[i][test_x_pred[i] != 1] = 0
                seedlist = []
                for i in range(test_x_true.shape[0]):
                    seed = np.where(test_x_pred[i]==1)
                    seed = np.array(seed)
                    seed = seed.tolist()
                    seed = seed[0]
                    seedlist.append(seed)
                Q = genraQ(g, seedlist,t)
                Q = torch.tensor(Q).to(device)
                p0 = nntheta(test_y, Q, test_x_hat, t)
                p0 = torch.nn.functional.gumbel_softmax(torch.log(p0), 0.000001, True)
                xt_1.append(p0[[kkk]].cpu().detach())
                test_y=p0
                test_y0 = torch.argmax(p0 , dim=2)


        test_x_pred=torch.argmax(test_y , dim=2)
        for i in range(test_x_true.shape[0]):
            test_x_pred[i][test_x_pred[i] ==2] = 0
        for i in range(test_x_true.shape[0]):

            test_precision_all += precision_score(test_x_true[i].cpu().detach().numpy(),
                                                  test_x_pred[i].cpu().detach().numpy().astype(np.int64),
                                                  zero_division=0)
            test_recall_all += recall_score(test_x_true[i].cpu().detach().numpy(),
                                            test_x_pred[i].cpu().detach().numpy().astype(np.int64), zero_division=0)
            f1 += f1_score(test_x_true[i].cpu().detach().numpy(),
                                            test_x_pred[i].cpu().detach().numpy().astype(np.int64), zero_division=0)
            roc_auc += roc_auc_score(test_x_true[i].cpu().detach().numpy(),
                                            test_x_pred[i].cpu().detach().numpy().astype(np.int64))


        print(
        "\ttest_Precision: {:.4f}".format(test_precision_all / test_sample_number),
        "\ttest_Recall: {:.4f}".format(test_recall_all / test_sample_number),
        "\ttest_F1: {:.4f}".format(f1 / test_sample_number),
        "\ttest_roc_auc: {:.4f}".format(roc_auc / test_sample_number),
        )
#
# def graph_print(xt,g):
#     '''
#     输入单个快照的感染数据
#     '''
#     color_map = []
#     for x in xt:
#         if x == 0:
#             color_map.append('turquoise')
#         elif x == 1:
#             color_map.append('lightcoral')
#         else:
#             color_map.append('palegreen')
#     f = open('F:/diffusion_source_detect/dataset/karate_pos.pkl', 'rb')
#     pos = pickle.load(f)
#     nx.draw(g, pos=pos,node_color=color_map,with_labels=True)
#     plt.show()
# # np.savez('F:/diffusion_source_detect/diff_dataset/xt_1.npz', xt_1)
# # for i,x in enumerate(xt_1):
# #     if i>0:
# #         xt_1[i]=torch.squeeze(x,dim=0)
# # xt_1=torch.stack(xt_1,0)
# # xt=torch.argmax(xt_1 , dim=2)
# #
# #
# # for t,x in enumerate(xt):          ####一个实例感染过程的可视化
# #     x=x.tolist()
# #     print(x)
# #     if (t+1)%4==0:
# #         graph_print(x,g)
# # print("*********************************")
# # for t,x in enumerate(gt):          ####一个实例感染过程的可视化
# #     print(x)
# #     if (t+1)%4==0:
# #         graph_print(x,g)
# np.savez('F:/diffusion_source_detect/diff_dataset/xt_1.npz', xt_1)
# np.savez('F:/diffusion_source_detect/diff_dataset/gt.npz', gt)
# xt=[]
# for t,x in enumerate(xt_1):
#     if t<2:
#         xt.append(torch.argmax(x , dim=1))
#     else:xt.append(torch.argmax(x[0] , dim=1))
#
# xt=torch.stack(xt,0)
# gt=torch.stack(gt,0)
# for i in range(xt.shape[0]):
#     xt[i][xt[i]==2]=0
#     gt[i][gt[i]==2]=0
#
#
#
# print("MSE:",MSE(xt,gt))