import numpy as np
import scipy.io as sio
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
import scipy
import torch
from sklearn.metrics import f1_score
import scipy.sparse as sp
import torch.nn.functional as F
from sklearn.decomposition import PCA
from warnings import filterwarnings
import torch.nn as nn

filterwarnings('ignore')


def load_network(file):
    net = sio.loadmat(file)
    x, a, y = net['attrb'], net['network'], net['group']
    if not isinstance(x, scipy.sparse.lil_matrix):
        x = lil_matrix(x)
    return a, x, y


def my_scale_sim_mat(w):
    """L1 row norm of a matrix"""
    rowsum = np.array(np.sum(w, axis=1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    w = r_mat_inv.dot(w)
    return w


def my_scale_sim_mat_torch(w):
    """L1 row norm of a matrix of torch version"""
    r = 1 / w.sum(1)
    r[r.isinf()] = 0
    return r.diag() @ w

def calculate_pred_label_t(pred_label, cluster_label):
    """calculate predicted label by comparing clf loss with pred label kmean"""
    # _, indices = torch.max(pred_logit_s, dim=1)
    # pred_logit_s = one_hot_encode_torch(indices, pred_logit_t.shape[1])
    # pred_logit_s = pred_logit_s.to(pred_logit_t.device)
    return pred_label * cluster_label
def calculate_centroid_2(label, emb):
    """calculate centroid of each class"""
    norm_Y = my_scale_sim_mat_torch(label.T)
    centroid = torch.mm(norm_Y, emb)
    return centroid
def one_hot_encode_torch(x, n_classes):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
     """
    x = x.type(torch.LongTensor)
    return torch.eye(n_classes)[x]


def mini_batch(x, y, a, n, batch_size):
    idx = list(range(x.shape[0]))
    np.random.shuffle(idx)
    n = np.ceil(n / batch_size).astype('int') * batch_size
    idx = (idx * (n // x.shape[0] + bool(n % x.shape[0])))[:n]
    x, y = x[idx], y[idx]
    a = a[idx][:, idx]

    for i in range(n // batch_size + bool(n % batch_size)):
        start, end = i * batch_size, (i + 1) * batch_size
        shuffle_index = idx[i * batch_size:(i + 1) * batch_size]
        yield x[start:end], y[start:end], a[start:end, start:end], shuffle_index


def f1_scores(y_pred, y_true):
    def predict(y_tru, y_pre):
        top_k_list = np.array(np.sum(y_tru, 1), np.int32)
        prediction = []
        for i in range(y_tru.shape[0]):
            pred_i = np.zeros(y_tru.shape[1])
            pred_i[np.argsort(y_pre[i, :])[-top_k_list[i]:]] = 1
            prediction.append(np.reshape(pred_i, (1, -1)))
        prediction = np.concatenate(prediction, axis=0)
        return np.array(prediction, np.int32)

    results = {}
    predictions = predict(y_true, y_pred)
    averages = ["micro", "macro"]
    for average in averages:
        results[average] = f1_score(y_true, predictions, average=average)
    return results["micro"], results["macro"]


def sim(z1: torch.Tensor, z2: torch.Tensor, hidden_norm: bool = True):
    if hidden_norm:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


#  点 - 点
def inter_view_nei_loss(z1: torch.Tensor, z2: torch.Tensor, tau, adj, hidden_norm: bool = True):
    adj[adj > 0] = 1

    f = lambda x: torch.exp(x / tau)
    between_sim = f(sim(z1, z2, hidden_norm))

    nei_count = torch.sum(adj, 1) + 1
    nei_count = torch.squeeze(torch.tensor(nei_count))

    loss = (between_sim.mul(adj)).sum(1) / between_sim.sum(1)
    loss = loss / nei_count
    loss[loss == 0] = 1

    # return -torch.log(loss + 1e-10)
    return -torch.log(loss)


def nei_dis_loss1(z1: torch.Tensor, z2: torch.Tensor, tau, adj, hidden_norm: bool = True):
    '''neighbor discrimination contrastive loss'''
    ###先求和再log
    # np.fill_diagonal(adj, 0) #remove self-loop
    adj = adj - torch.diag_embed(adj.diag())  # remove self-loop
    adj[adj > 0] = 1
    # nei_count=np.sum(adj,1)*2+1 ###intra-view nei+inter-view nei+self inter-view
    nei_count = torch.sum(adj, 1) * 2 + 1  ###intra-view nei+inter-view nei+self inter-view
    nei_count = torch.squeeze(torch.tensor(nei_count))
    # adj = torch.tensor(adj)

    f = lambda x: torch.exp(x / tau)
    refl_sim = f(sim(z1, z1, hidden_norm))
    between_sim = f(sim(z1, z2, hidden_norm))

    loss = (between_sim.diag() + (refl_sim.mul(adj)).sum(1) + (between_sim.mul(adj)).sum(1)) / (
            refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())
    loss = loss / nei_count  ###divided by the number of positive pairs for each node

    return -torch.log(loss)


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, adj,
                     mean: bool = True, tau: float = 1.0, hidden_norm: bool = True):
    h1 = z1
    h2 = z2

    l1 = nei_dis_loss1(h1, h2, tau, adj, hidden_norm)
    l2 = nei_dis_loss1(h2, h1, tau, adj, hidden_norm)

    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()
    return ret


def multihead_contrastive_loss(heads, adj, tau: float = 1.0):
    ###算每个head到第一个head
    loss = torch.tensor(0, dtype=float, requires_grad=True)
    for i in range(1, len(heads)):
        loss = loss + contrastive_loss(heads[0], heads[i], adj, tau=tau)
    return loss / (len(heads) - 1)


def class_class_inter_view_nei_loss_2(z1: torch.Tensor, z2: torch.Tensor, tau, O_inter, O_intra,
                                      hidden_norm: bool = True):
    """class-class inter view neighbor contrastive loss"""
    O_intra = O_intra - (O_intra.diag().diag())

    f = lambda x: torch.exp(x / tau)

    inter_sim_st = f(sim(z1, z2, hidden_norm))
    intra_sim_ss = f(sim(z1, z1, hidden_norm))

    # fenzi
    molecule = inter_sim_st.diag() * O_inter.diag()
    # fenmu
    denominator = (inter_sim_st * O_inter).sum(1) + (intra_sim_ss * O_intra).sum(1)
    # 分母为0，会导致nan
    denominator[denominator == 0] = 1
    loss = molecule / denominator
    loss[loss == 0] = 1
    # loss[loss.isnan()] = 1
    return -torch.log(loss)

#  点 类
def inter_view_nei_loss_NC(z1: torch.Tensor, z2: torch.Tensor, tau, label, hidden_norm: bool = True):
    """node-class inter view neighbor contrastive loss"""

    f = lambda x: torch.exp(x / tau)
    inter_sim_st = f(sim(z1, z2, hidden_norm))
    # 分子 正样本
    molecule = inter_sim_st * label
    molecule = molecule.sum(1)
    # 分母
    denominator = inter_sim_st.sum(1)
    # 分母为0，会导致nan
    denominator[denominator == 0] = 1
    loss = molecule / denominator
    loss[loss == 0] = 1
    # loss[loss.isnan()] = 1
    return -torch.log(loss)


