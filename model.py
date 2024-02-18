from torch import nn
import torch
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv



        
        
class GradReverse(torch.autograd.Function):
    # lambd = 0.0

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return args[0].view_as(args[0])

    @staticmethod
    def backward(ctx, *grad_outputs):
        return grad_outputs[0]*-GradReverse.lambd


def grad_reverse(x, lambd=1.0):
    GradReverse.lambd = lambd
    return GradReverse.apply(x)


class GAT(nn.Module):
    def __init__(self, num_layers, in_dim, num_hidden, heads, activation, feat_drop, attn_drop, negative_slope, residual, num_classes):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.gat_layers = nn.ModuleList()
        self.activation = activation

        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l - 1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, False, self.activation))
        # # # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs, g):
        heads = []
        h = inputs
        # get hidden_representation
        for l in range(self.num_layers):
            temp = h.flatten(1)  # 保存上一层multi-head flatten拼接的结果
            # h = self.gat_layers[l](self.g, temp)
            h = self.gat_layers[l](g, temp,  get_attention=True)[0]
            # h = self.gat_layers[l](self.g, h).flatten(1)
        # get heads
        for i in range(h.shape[1]):
            heads.append(h[:, i])
        # output projection
        logits = self.gat_layers[-1](g, h.flatten(1)).mean(1)
        # logits = self.gat_layers[-1](g, torch.cat(h_allLayers, axis=1)).mean(1)
        # hidden_rep=h.flatten(1)
        return heads, logits




class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim_mlp):
        super(DomainDiscriminator, self).__init__()
        self.h_dann_1 = nn.Linear(input_dim_mlp, 32)
        self.h_dann_2 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, 2)
        # std = 1/(input_dim_mlp/2)**0.5
        # nn.init.trunc_normal_(self.h_dann_1.weight, std=std, a=-2*std, b=2*std)
        nn.init.xavier_normal_(self.h_dann_1.weight, 1.414)
        nn.init.constant_(self.h_dann_1.bias, 0.1)
        # nn.init.trunc_normal_(self.h_dann_2.weight, std=0.125, a=-0.25, b=0.25)
        nn.init.xavier_normal_(self.h_dann_2.weight, 1.414)
        nn.init.constant_(self.h_dann_2.bias, 0.1)
        # nn.init.trunc_normal_(self.output_layer.weight, std=0.125, a=-0.25, b=0.25)
        nn.init.xavier_normal_(self.output_layer.weight, 1.414)
        nn.init.constant_(self.output_layer.bias, 0.1)

    def forward(self, h_grl):
        h_grl = F.relu(self.h_dann_1(h_grl))
        h_grl = F.relu(self.h_dann_2(h_grl))
        d_logit = self.output_layer(h_grl)
        return d_logit




class MAGCL(nn.Module):
    def __init__(self, num_layers, in_dim, num_hidden, heads, activation, feat_drop, attn_drop, negative_slope, residual, num_classes, input_dim_mlp):
        super(ACDNE, self).__init__()
        self.network_embedding = GAT(num_layers, in_dim, num_hidden, heads, activation, feat_drop, attn_drop, negative_slope, residual, num_classes)
        self.domain_discriminator = DomainDiscriminator(input_dim_mlp)

    def forward(self, features_s, features_t, g_s, g_t,grl_lambda):
        head_s,pred_logit_s = self.network_embedding(features_s, g_s)
        emb_s = torch.cat(head_s, axis=1)
        head_t,pred_logit_t= self.network_embedding(features_t, g_t)
        emb_t = torch.cat(head_t, axis=1)      
        
        emb = torch.cat((emb_s, emb_t), 0)

        
        # Domain_Discriminator
        h_grl = grad_reverse(emb, grl_lambda)
        d_logit = self.domain_discriminator(h_grl)

        return pred_logit_s,  pred_logit_t, d_logit, emb_s, emb_t, head_s, head_t 



