import argparse
import random
import time

from torch import nn
from model import ACDNE
import numpy as np
import torch
from scipy.sparse import vstack
import dgl
from scipy.sparse import lil_matrix
from utils import *
from sklearn.cluster import KMeans

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0, help="which GPU to use. Set -1 to use CPU.")
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr-ini', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--l2-w', type=float, default=0.01, help='weight of L2-norm regularization')
parser.add_argument("--num-heads", type=int, default=4, help="number of hidden attention heads")
parser.add_argument("--num-layers", type=int, default=4, help="number of hidden layers")
parser.add_argument("--num-hidden", type=int, default=16, help="number of hidden units")
parser.add_argument("--intra_view_gcl_wei", type=float, default=1, help="weight of network-specific GCL")
parser.add_argument("--inter_view_gcl_wei", type=float, default=1, help="weight of cross-network GCL")
parser.add_argument("--in-drop", type=float, default=0.3, help="input feature dropout")
parser.add_argument("--attn-drop", type=float, default=0.4, help="attention dropout")
parser.add_argument("--tau", type=float, default=1, help="temperature-scales")
parser.add_argument('--grl-weight', type=int, default=1, help="--lr-weight")
parser.add_argument("--num-out-heads", type=int, default=2, help="number of output attention heads")
parser.add_argument("--batch_size", type=int, default=4000, help="batch_size for each domain")
parser.add_argument("--threashold", type=float, default=0.1, help="remove nodes whose similarity less than 0.3")
parser.add_argument("--threshold_pos", type=float, default=0.9, help="remove nodes whose similarity less than 0.8")
parser.add_argument("--threshold_neg", type=float, default=0.1, help="remove nodes whose similarity less than 0.2")
parser.add_argument('--data_src', type=str, default='citationv1', help='source dataset name')
parser.add_argument('--data_trg', type=str, default='dblpv7', help='target dataset name')

args = parser.parse_args()


if args.gpu != -1 and torch.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

source = args.data_src
target = args.data_trg
emb_filename = str(source) + '_' + str(target)
f = open('./output/' + emb_filename + '.txt', 'a')
f.write('{}\n'.format(args))
f.flush()
# Load source data
A_s, X_s, Y_s = load_network('./input/' + str(source) + '.mat')
num_feat = X_s.shape[1]
num_class = Y_s.shape[1]
num_nodes_s = X_s.shape[0]
g_s = dgl.from_scipy(A_s).to(args.device)
g_s = dgl.remove_self_loop(g_s)
g_s = dgl.add_self_loop(g_s)
# Load target data
A_t, X_t, Y_t = load_network('./input/' + str(target) + '.mat')
num_nodes_t = X_t.shape[0]
g_t = dgl.from_scipy(A_t).to(args.device)
g_t = dgl.remove_self_loop(g_t)
g_t = dgl.add_self_loop(g_t)

features_s = torch.Tensor(X_s.todense()).to(args.device)
features_t = torch.Tensor(X_t.todense()).to(args.device)

Y_s_tensor = torch.LongTensor(Y_s).to(args.device)


random_state = 0
heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
mlp_input_dim = int((args.num_hidden) * (args.num_heads))
ST_max = max(X_s.shape[0], X_t.shape[0])


microAllRandom = []
macroAllRandom = []
best_microAllRandom = []
best_macroAllRandom = []
numRandom = 5


for random_state in range(numRandom):
    random_state = random_state + 1
    print('%d-th random split' % (random_state))

    random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state) if torch.cuda.is_available() else None
    np.random.seed(random_state)

    clf_type = 'multi-label'

    model = MAGCL(
        num_layers=args.num_layers,
        in_dim=num_feat,
        num_hidden=args.num_hidden,
        heads=heads,
        activation=F.elu,
        feat_drop=args.in_drop,
        attn_drop=args.attn_drop,
        negative_slope=0.2,
        residual=False,
        num_classes=num_class,
        input_dim_mlp=mlp_input_dim
    )

    model = model.to(args.device)

    clf_loss_f = nn.BCEWithLogitsLoss(reduction='none') if clf_type == 'multi-label' \
        else nn.CrossEntropyLoss()

    domain_loss_f = nn.CrossEntropyLoss()

    t_total = time.time()
    domain_loss_all = []
    total_loss_all = []

    best_epoch = 0
    best_micro_f1 = 0
    best_macro_f1 = 0

    pred_Y_t = np.zeros(Y_t.shape)
    dur = []
    for epoch in range(args.epochs):

        t = time.time()

        for batch_idx, (batch_s, batch_t) in enumerate(
                zip(mini_batch(X_s, Y_s, A_s, ST_max, args.batch_size),
                    mini_batch(X_t, pred_Y_t, A_t, ST_max, args.batch_size))):

            feat_s, label_s, adj_s, shuffle_index_s = batch_s
            feat_t, pred_label_t, adj_t, shuffle_index_t = batch_t


            feat_s = torch.FloatTensor(feat_s.toarray()).to(args.device)
            label_s = torch.LongTensor(label_s).to(args.device)
            adj_s = torch.FloatTensor(adj_s.toarray()).to(args.device)

            feat_t = torch.FloatTensor(feat_t.toarray()).to(args.device)
            pred_label_t = torch.FloatTensor(pred_label_t).to(args.device)
            adj_t = torch.FloatTensor(adj_t.toarray()).to(args.device)

            g_s1 = dgl.from_scipy(sp.coo_matrix(adj_s.cpu())).to(args.device)
            g_s1 = dgl.remove_self_loop(g_s1)
            g_s1 = dgl.add_self_loop(g_s1)

            g_t1 = dgl.from_scipy(sp.coo_matrix(adj_t.cpu())).to(args.device)
            g_t1 = dgl.remove_self_loop(g_t1)
            g_t1 = dgl.add_self_loop(g_t1)

            p = float(epoch) / args.epochs
            lr = args.lr_ini / (1. + 10 * p) ** 0.75
            grl_lambda = 2. / (1. + np.exp(-10. * p)) - 1  # gradually change from 0 to 1

            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=args.l2_w)
            optimizer.zero_grad()

            pred_logit_s, pred_logit_t, d_logit, emb_s, emb_t, head_s, head_t = model(feat_s, feat_t, g_s1, g_t1,
                                                                                      grl_lambda)

            if clf_type == 'multi-class':
                clf_loss = clf_loss_f(pred_logit_s, torch.argmax((label_s), 1))
            else:
                clf_loss = clf_loss_f(pred_logit_s, (label_s).float())
                clf_loss = torch.sum(clf_loss) / (label_s).shape[0]

            domain_label = np.vstack([np.tile([1., 0.], [feat_s.shape[0], 1]), np.tile([0., 1.], [feat_t.shape[0], 1])])
            domain_loss = domain_loss_f(d_logit, torch.argmax(torch.FloatTensor(domain_label).to(args.device), 1))

            _, indices = torch.max(pred_logit_t,
                                   dim=1)

            ### network-specific graph contrastive loss
            intra_view_s = multihead_contrastive_loss(head_s, adj_s, args.tau)
            intra_view_t = multihead_contrastive_loss(head_t, adj_t, args.tau)
            within_network_contrastive_loss = intra_view_s + intra_view_t

            # ### cross-network graph contrastive loss
            comlabel_inter_st = torch.mm(label_s.float(),
                                         pred_label_t.t())  ##check whether two nodes in s and t have same label
            comlabel_inter_ts = torch.mm(pred_label_t,
                                         label_s.float().t())  ##check whether two nodes in s and t have same label
            cross_view_st = inter_view_nei_loss(emb_s, emb_t, args.tau, comlabel_inter_st)
            cross_view_ts = inter_view_nei_loss(emb_t, emb_s, args.tau, comlabel_inter_ts)

            cross_network_constrastive_loss = (cross_view_st + cross_view_ts) * 0.5
            cross_network_constrastive_loss = cross_network_constrastive_loss.mean()

            total_loss = clf_loss + domain_loss + within_network_contrastive_loss + cross_network_constrastive_loss



            total_loss.backward()
            total_loss_all.append(total_loss.item())
            optimizer.step()


        '''Compute evaluation on test data by the end of each epoch'''
        model.eval()  # deactivates dropout during validation run.
        with torch.no_grad():
            pred_logit_s, pred_logit_t, d_logit, emb_s, emb_t, _, _ = model(features_s, features_t, g_s, g_t, 1)

            if clf_type == 'multi-class':
                clf_loss = clf_loss_f(pred_logit_s, torch.argmax((Y_s_tensor), 1))
            else:
                clf_loss = clf_loss_f(pred_logit_s, (Y_s_tensor).float())
                clf_loss = torch.sum(clf_loss) / (Y_s_tensor).shape[0]
            domain_label = np.vstack(
                [np.tile([1., 0.], [features_s.shape[0], 1]), np.tile([0., 1.], [features_t.shape[0], 1])])
            domain_loss = domain_loss_f(d_logit, torch.argmax(torch.FloatTensor(domain_label).to(args.device), 1))
            domain_loss_all.append(domain_loss.item())

            pred_prob_xs = F.sigmoid(pred_logit_s) if clf_type == 'multi-label' else F.softmax(pred_logit_s)
            pred_prob_xt = F.sigmoid(pred_logit_t) if clf_type == 'multi-label' else F.softmax(pred_logit_t)
            f1_s = f1_scores(pred_prob_xs.cpu(), Y_s)
            print('epoch %d: Source micro-F1: %f, macro-F1: %f' % (epoch, f1_s[0], f1_s[1]))
            f1_t = f1_scores(pred_prob_xt.cpu(), Y_t)
            print('epoch %d: Target testing micro-F1: %f, macro-F1: %f' % (epoch, f1_t[0], f1_t[1]))
            print('domain_loss: %f, clf_loss: %f' % (domain_loss, clf_loss))

            ###generate pseduo-label for target nodes by k-means
            norm_Y_s = my_scale_sim_mat(Y_s.T)
            norm_Y_s = torch.FloatTensor(norm_Y_s)
            emb_s_class = torch.mm(norm_Y_s, emb_s.cpu())  ###c*d, each row represents avg rep for one class
            kmeans = KMeans(n_clusters=num_class, init=emb_s_class, random_state=0).fit(emb_s.cpu())
            pred_Y_t = kmeans.predict(emb_t.cpu())
            pred_Y_t = np.eye(num_class)[pred_Y_t]

            # # ###remove noisy nodes, far away from the cluster centroid
            X_dist = kmeans.transform(emb_t.cpu()) ** 2
            sim = np.exp(-X_dist)  ###convert distance to similarity
            pred_Y_t_noisy = np.where(sim > args.threashold, 1, 0)  ## remove nodes whose similarity less than 0.3
            pred_Y_t = np.multiply(pred_Y_t, pred_Y_t_noisy)

            # pred_Y_t =pred_Y_t +np.matmul(my_scale_sim_mat(csc_matrix.toarray(A_t)), pred_Y_t)
            # pred_Y_t[pred_Y_t>1]=1

            if f1_t[1] > best_macro_f1:
                best_micro_f1 = f1_t[0]
                best_macro_f1 = f1_t[1]
                best_epoch = epoch

    print('Target best epoch %d, micro-F1: %f, macro-F1: %f' % (best_epoch, best_micro_f1, best_macro_f1))

    microAllRandom.append(float(f1_t[0]))
    macroAllRandom.append(float(f1_t[1]))
    best_microAllRandom.append(float(best_micro_f1))
    best_macroAllRandom.append(float(best_macro_f1))

'''avg F1 scores over 5 random splits'''
micro = np.mean(microAllRandom)
macro = np.mean(macroAllRandom)
micro_sd = np.std(microAllRandom)
macro_sd = np.std(macroAllRandom)

'''avg best F1 scores over 5 random splits'''
best_micro = np.mean(best_microAllRandom)
best_macro = np.mean(best_macroAllRandom)
best_micro_sd = np.std(best_microAllRandom)
best_macro_sd = np.std(best_macroAllRandom)
print(
    "The avergae best micro and macro F1 scores over {} random initializations are:  {} +/- {} and {} +/- {}: ".format(
        numRandom, best_micro, best_micro_sd, best_macro, best_macro_sd))
f.write(
    "The avergae best micro and macro F1 scores over {} random initializations are:  {} +/- {} and {} +/- {}: \n".format(
        numRandom, best_micro, best_micro_sd, best_macro, best_macro_sd))

f.flush()
f.close()
