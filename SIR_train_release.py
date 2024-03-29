import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
import random
from sklearn import model_selection
from model import GCNB,LRScheduler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
from torch.optim import Adam
import warnings

warnings.filterwarnings("ignore")

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


print('Is GPU available? {}\n'.format(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def npz2list(path):
    list1 = np.load(path,allow_pickle=True)
    arr_0 = list1['arr_0']
    return arr_0

from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import platform
import ctypes
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def creat_Pt(iterations, beta, gamma, m, g,  infected_nodes,n):
    MAX_NODES = 20000
    if platform.system().lower() == 'windows':
        mydll = ctypes.CDLL(r'C:\Users\Administrator\Desktop\SIM\NMLGB\Dll1\x64\Release\Dll1.dll', winmode=0)
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
    infected_nodes_c = (ctypes.c_int * MAX_NODES)(*infected_nodes)
    num_infected_nodes=len(infected_nodes)
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

def loss_all(x, x_hat, y_hat):
    monotone_loss = torch.sum(torch.relu(y_hat-y_hat[0]))
    reproduction_loss = F.binary_cross_entropy(x_hat, x, reduction='mean')
    total_loss = reproduction_loss+monotone_loss
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

def Q_mbmm(Q1,Q2):
    "Q1 Q2并行乘法"
    Q1=torch.tensor(Q1,dtype=torch.float)
    Q2 = torch.tensor(Q2, dtype=torch.float)
    Q=torch.zeros_like(Q2)
    for i,q in enumerate(Q1):
        Q[i]=torch.bmm(q,Q2[i])
    return Q

def Qtba(Q,t):
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
    hamada=distru1*distru2 #50x34x3
    q_pri = normalize_tensor(hamada).to(device)
    return q_pri
def genQp(x_pred,t):
    seedlist = []
    x_pred=x_pred.cpu().detach()
    for i in range(x_pred.shape[0]):
        seed = np.where(x_pred[i] == 1)
        seed = np.array(seed)
        seed = seed.tolist()
        seed = seed[0]
        seedlist.append(seed)
    Q=genraQ(g, seedlist, t)
    return torch.tensor(Q).to(device)
def PandQ_genq(x,y, Q, x_pred, t):

    Qp=genQp(x_pred,t)
    x = torch.nn.functional.one_hot(x.to(torch.int64), num_classes=3).to(device)
    distru1 = xt_X_QTrans(y, Q[:, t - 1, :, :, :, ]).to(device)
    distrup1= xt_X_QTrans(y, Qp[:, t - 1, :, :, :, ]).to(device)
    Qba_t_1 = Qtba(Q, t - 2)
    Qbap_t_1=Qtba(Qp, t - 2)
    Qba_t_1 = torch.tensor(Qba_t_1).to(device)
    Qbap_t_1 = torch.tensor(Qbap_t_1).to(device)
    distru2 = xt_X_Q(x, Qba_t_1).to(device)
    hamada = distru1 * distru2  # 50x34x3
    q_pri = normalize_tensor(hamada).to(device)
    q_xt_xt_1=distrup1   #50x34x3
    xs,xi=nnout(x_pred)     #50x34
    ps=torch.tensor([1,0,0])
    x0_s=tensormul(ps,xs.shape[1]).to(device)
    x0_s=tensormul(x0_s,xs.shape[0]).to(device) #50X34x3
    pi=torch.tensor([0,1,0]).to(device)
    x0_i=tensormul(pi,xi.shape[1]).to(device)
    x0_i = tensormul(x0_i, xi.shape[0]).to(device)
    q_ps=xt_X_Q(x0_s, Qbap_t_1) .to(device)   #50x34x3
    q_pi=xt_X_Q(x0_i, Qbap_t_1).to(device)
    xs,xi=torch.unsqueeze(xs,2),torch.unsqueeze(xi,2)
    rs,ri=torch.zeros_like(q_ps),torch.zeros_like(q_ps)
    for i,q in enumerate(q_ps):
        a = q * xs[i]
        rs[i]=a
    for i,q in enumerate(q_pi):
        ri[i]=q*xi[i]
    ptheta=ri+rs#50x34x3
    ptheta=ptheta.to(device)
    phamada=q_xt_xt_1*ptheta.to(device)
    p_pri=normalize_tensor(phamada).to(device)
    return q_pri,p_pri

def infecting_rule_loss(pred,prdata,Adj):
    Adj = torch.tensor(Adj).to(device)
    Adj=Adj+ torch.eye(len(Adj)).to(device)
    prdata=torch.argmax(prdata, dim=2)
    pred=torch.argmax(pred, dim=2)

    for i in range(pred.shape[0]):
        pred[i][pred[i]==2]=0
        prdata[i][prdata[i]==2]=0
    reg=0
    pred=pred.to(torch.float)
    prdata=prdata.to(torch.float)
    for i in range(pred.shape[0]):
        reg=reg+ torch.mean(torch.nn.functional.relu(prdata[i] - torch.matmul(Adj, pred[i]))).to(torch.float)


    return reg/pred.shape[0]
def Lvb_loss(x,y, Q, x_pred, t,adj):
    adj=torch.tensor(adj,dtype=torch.float).to(device)
    if t>0:
        p0,pr=PandQ_genq(x,y, Q, x_pred, t)
        kl_loss = F.kl_div(p0.softmax(dim=2).log(), pr.softmax(dim=2), reduction='mean')
        x_t_1=torch.nn.functional.gumbel_softmax(torch.log(p0), 0.0001, True)
        ruleloss=infecting_rule_loss(x_t_1,y,adj)

    else:
        x_pred=x_pred.to(device)
        x = x.to(device)
        kl_loss=F.binary_cross_entropy(x_pred,x, reduction='mean')
        ruleloss=torch.tensor([0],dtype=torch.float).to(device)
    return kl_loss,ruleloss
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
def genraQ(g,seedlist,t):
    infect_rate,recover_rate,t=0.03,0.015,t+1
    Qt=[]
    qt=[]
    for i,seed in enumerate(seedlist):
        prob=creat_Pt(t, infect_rate,recover_rate, 10000, g, seed,12)
        q=pt2Qt(prob)
        qt.append(q)
    return qt

MSE=torch.nn.MSELoss(reduction='mean')

def train(train_set,Qtrain_set,val_set,model,epoch,tmax):
    optimizer = Adam(model.parameters(), lr=(2e-3))
    lr_scheduler = LRScheduler(optimizer)
    model = model.to(device)
    sample_number = train_set[:].shape[0] * train_set[:].shape[1]
    val_sample_number = val_set[:].shape[0] * val_set[:].shape[1]
    loss_record = []
    scheduler = MultiStepLR(optimizer, milestones=[500,1200], gamma=0.97)
    for epoch in range(epoch):
        model.train()
        re_overall = 0
        total_overall = 0
        precision_all = 0
        recall_all = 0
        rule_overall=0
        for batch_idx, data_pair in tqdm(enumerate(train_set)):
            data_pair=torch.tensor(data_pair).to(device)
            Q=Qtrain_set[batch_idx]
            Q=torch.tensor(Q,dtype=torch.float).to(device)
            t_list=torch.range(0,tmax)
            sum_A = sum(t_list)
            t = random.choices(t_list, weights=[(a+1) / (sum_A+t_list.shape[0]) for a in t_list])[0]
            t=int(t)
            x = data_pair[:, :, 0].float().to(device)   #x[0]是所有x[i]的超集
            y = data_pair[:, :,t].float().to(device)
            y=torch.nn.functional.one_hot(y.to(torch.int64), num_classes=3).to(device)
            optimizer.zero_grad()
            x_true = x.cpu().detach()
            x_hat = model(g0,y,t).to(device)
            y_hat =torch.zeros_like(x_hat)
            re_loss,  loss = loss_all(x, x_hat,  y_hat)
            x_pred = x_hat.cpu().detach()
            kl_loss,rule_loss = Lvb_loss(x,y, Q, x_hat, t,adj)
            loss=loss+kl_loss+rule_loss
            rule_overall +=rule_loss.item()*x_hat.size(0)
            re_overall +=kl_loss.item()*x_hat.size(0)
            total_overall += loss.item()*x_hat.size(0)
            for i in range(x_true.shape[0]):
                x_pred[i][x_pred[i] > 0.5] = 1
                x_pred[i][x_pred[i] != 1] = 0
                precision_all += precision_score(x_true[i].cpu().detach().numpy(), x_pred[i].cpu().detach().numpy(), zero_division=0)
                recall_all += recall_score(x_true[i].cpu().detach().numpy(), x_pred[i].cpu().detach().numpy(), zero_division=0)
            loss.backward()
            optimizer.step()
        loss_record.append(total_overall/ sample_number)


        print("Epoch: {}".format(epoch+1),

              "\tRule: {:.4f}".format(rule_overall / sample_number),
              "\tLvb: {:.4f}".format(re_overall / sample_number),
              "\tTotal: {:.4f}".format(total_overall / sample_number),
              "\tprecision_all: {:.4f}".format(precision_all / sample_number),
              "\trecall_all: {:.4f}".format(recall_all / sample_number),
             )
        val_precision_all = 0
        val_recall_all = 0
        val_f1=0
        val_auc=0
        mse=0
        if (epoch+1)%1==0:
            model.eval()
            val_loss=[]
            for val_batch_idx, val_data_pair in enumerate(val_set):
                val_data_pair = torch.tensor(val_data_pair).to(device)
                with torch.no_grad():
                    val_x = val_data_pair[:, :, 0].float().to(device)  # x[0]是所有x[i]的超集
                    val_y = val_data_pair[:, :, tmax].to(device)
                    val_y = torch.nn.functional.one_hot(val_y.to(torch.int64), num_classes=3)
                    val_x_true = val_x.cpu().detach()
                    val_x_hat = model(g0,val_y,tmax).to(device)
                    val_x_pred = val_x_hat.cpu().detach()
                    re_loss, valloss = loss_all(val_x_true, val_x_pred, y_hat)
                    val_loss.append(valloss.cpu())
                    for i in range(val_x_true.shape[0]):
                        val_x_pred[i][val_x_pred[i] > 0.4] = 1
                        val_x_pred[i][val_x_pred[i] != 1] = 0
                        val_precision_all += precision_score(val_x_true[i].cpu().detach().numpy(),
                                                              val_x_pred[i].cpu().detach().numpy().astype(np.int64),
                                                              zero_division=0)
                        val_recall_all += recall_score(val_x_true[i].cpu().detach().numpy(),
                                                        val_x_pred[i].cpu().detach().numpy().astype(np.int64), zero_division=0)
                        val_f1 += f1_score(val_x_true[i].cpu().detach().numpy(),
                                                        val_x_pred[i].cpu().detach().numpy().astype(np.int64), zero_division=0)
                        val_auc += roc_auc_score(val_x_true[i].cpu().detach().numpy(),
                                                        val_x_pred[i].cpu().detach().numpy().astype(np.int64))
                        val_x_pred[i][val_x_pred[i]==2] = 0
                        val_x_true[i][val_x_true[i] == 2] = 0
                        mse+=MSE(val_x_pred[i], val_x_true[i])/tmax
            val_loss=np.mean(val_loss)
            # lr_scheduler(val_loss)
            scheduler.step()

            print("\tVal_loss: {:.4f}".format(val_loss),
                "\tLearning rate: {:.8f}".format(optimizer.param_groups[0]['lr']),
                "\tval_Precision: {:.4f}".format(val_precision_all / val_sample_number),
                "\tval_Recall: {:.4f}".format(val_recall_all / val_sample_number),
                "\tval_F1: {:.4f}".format(val_f1 / val_sample_number),
                "\tval_roc_auc: {:.4f}".format(val_auc / val_sample_number),
                "\tmse: {:.4f}".format(mse),
            )


        if (epoch+1)%1==0:
            PATH = ("./model/SIRnet.ckpt")
            torch.save(model.state_dict(), PATH)

    import matplotlib.pyplot as plt

    # plt.plot(loss_record)
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.show()
    PATH=("./model/SIRnet.ckpt")
    torch.save(model.state_dict(), PATH)


def cor():
    filename=r"F:\diffusion_source_detect\diff_dataset\twin\out.txt"
    g= nx.Graph()
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

g0=torch.tensor(nx.adjacency_matrix(g).todense(),dtype=torch.float).to(device)
model=GCNB(g0)


Q=npz2list('./dataset/karate/Qt.npz')
inverse_pairs=npz2list('./dataset/karate/trainset.npz')
inverse_pairs=torch.Tensor(inverse_pairs)
inverse_pairs=inverse_pairs.permute(0,1,3,2)
inverse_pairs=inverse_pairs[:,:,:,:]
print(inverse_pairs.shape)
train_set, rest_set,Qtrain_set,Qrest=model_selection.train_test_split(inverse_pairs,Q,train_size=0.8,test_size= 0.2,shuffle=True,random_state=0)
test_set,val_set=model_selection.train_test_split(rest_set,test_size= 0.5,shuffle=True,random_state=0)

train(train_set,Qtrain_set, test_set, model, 1600,tmax=20)
