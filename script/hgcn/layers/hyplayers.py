"""Hyperbolic layers."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, softmax, add_self_loops
from torch_scatter import scatter, scatter_add
from torch_geometric.nn.conv import MessagePassing, GATConv
from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import glorot, zeros
from script.hgcn.manifolds import PoincareBall, Hyperboloid
from torch_geometric.utils import to_dense_adj
import itertools
import geotorch




class HMPGCNConv(nn.Module):
    def __init__(self, manifold, in_features, out_features, c_in=1.0, c_out=1.0, dropout=0.0, act=F.leaky_relu,
                 use_bias=True):
        super(HMPGCNConv, self).__init__()
        self.c_in = c_in
        self.linear = HypMPLinear(manifold, in_features, out_features, c_in, dropout=dropout)
        self.agg = HypMPAgg(manifold, c_in)
        self.hyp_act = HypMPAct(manifold, c_out, act)
        self.manifold = manifold

    def forward(self, x, edge_index):
        h = self.linear.forward(x)
        h = self.agg.forward(h, edge_index)
        h = self.hyp_act.forward(h)
        return h


class HypGRU(nn.Module):
    def __init__(self, args, c):
        super(HypGRU, self).__init__()
        self.manifold = PoincareBall()
        self.c = c
        self.nhid = args.nhid
        self.weight_ih = Parameter(torch.Tensor(3 * args.nhid, args.nhid), requires_grad=True)
        self.weight_hh = Parameter(torch.Tensor(3 * args.nhid, args.nhid), requires_grad=True)
        self.tanh = nn.Tanh()
        if args.bias:
            self.bias = nn.Parameter(torch.ones(3, args.nhid) * 1e-5, requires_grad=False)
        else:
            self.register_buffer("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.nhid)
        for weight in itertools.chain.from_iterable([self.weight_ih, self.weight_hh]):
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, hyperx, hyperh):
        out = self.mobius_gru_cell(hyperx, hyperh, self.weight_ih, self.weight_hh, self.bias)
        return out

    def mobius_gru_cell(self, input, hx, weight_ih, weight_hh, bias, nonlin=None):
        W_ir, W_ih, W_iz = weight_ih.chunk(3)
        b_r, b_h, b_z = bias
        W_hr, W_hh, W_hz = weight_hh.chunk(3)

        z_t = self.manifold.logmap0(self.one_rnn_transform(W_hz, hx, W_iz, input, b_z), self.c).sigmoid()
        r_t = self.manifold.logmap0(self.one_rnn_transform(W_hr, hx, W_ir, input, b_r), self.c).sigmoid()

        rh_t = self.manifold.mobius_pointwise_mul(r_t, hx, c=self.c)


        h_tilde = self.tanh(self.one_rnn_transform(W_hh, rh_t, W_ih, input, b_h)) # tanh

        delta_h = self.manifold.mobius_add(-hx, h_tilde, c=self.c)
        zdelta = self.manifold.mobius_pointwise_mul(z_t, delta_h, c=self.c)
        h_out = self.manifold.mobius_add(hx, zdelta, c=self.c)
        return h_out

    def one_rnn_transform(self, W, h, U, x, b):
        W_otimes_h = self.manifold.mobius_matvec(W, h, self.c)
        U_otimes_x = self.manifold.mobius_matvec(U, x, self.c)
        Wh_plus_Ux = self.manifold.mobius_add(W_otimes_h, U_otimes_x, self.c)
        return self.manifold.proj(self.manifold.mobius_add(Wh_plus_Ux, b, self.c), self.c)

    def mobius_linear(self, input, weight, bias=None, hyperbolic_input=True, hyperbolic_bias=True, nonlin=None):
        if hyperbolic_input:
            output = self.manifold.mobius_matvec(weight, input)
        else:
            output = torch.nn.functional.linear(input, weight)
            output = self.manifold.expmap0(output)
        if bias is not None:
            if not hyperbolic_bias:
                bias = self.manifold.expmap0(bias)
            output = self.manifold.mobius_add(output, bias)
        if nonlin is not None:
            output = self.manifold.mobius_fn_apply(nonlin, output)
        output = self.manifold.project(output)
        return output

class HypMPLinear(nn.Module):
    """
    Hyperbolic (no tangent) linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout=0.0, use_bias=True):
        super(HypMPLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.p = dropout
        self.dropout = nn.Dropout(self.p)
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(1, out_features), requires_grad=True)
        self.weight = nn.Linear(in_features, out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        geotorch.orthogonal(self.weight, "weight")
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, x):
        if self.p > 0.0:
            res = self.manifold.proj(self.dropout(self.weight(x)), self.c)
        else:
            res = self.manifold.proj(self.weight(x), self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias, self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypMPAct(Module):
    """
    Hyperbolic (no tangent) activation layer.
    """

    def __init__(self, manifold, c, act):
        super(HypMPAct, self).__init__()
        self.manifold = manifold
        self.c = c
        self.act = act

    def forward(self, x):
        xt = self.act(x)
        return xt

    def extra_repr(self):
        return 'c={}'.format(
            self.c
        )

class HypMPAgg(MessagePassing):
    """
    Hyperbolic aggregation layer using degree.
    """

    def __init__(self, manifold, c):
        super(HypMPAgg, self).__init__()
        self.manifold = manifold
        self.c = torch.tensor(c)


    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index):

        edge_index, norm = self.norm(edge_index, x.size(0), dtype=x.dtype)
        s = self.manifold.p2k(x, self.c)
        node_i = edge_index[0]
        node_j = edge_index[1]
        lamb = self.manifold.lorenz_factor(s, keepdim=True)
        lamb = torch.nn.functional.embedding(node_j, lamb)
        norm = norm.view(-1, 1) # len(node_j) x 1
        support_w = norm * lamb # len(node_j) x 1
        s_j = torch.nn.functional.embedding(node_j, s)
        s_j = support_w * s_j
        tmp = scatter(support_w, node_i, dim=0, dim_size=x.size(0))
        s_out = scatter(s_j, node_i, dim=0, dim_size=x.size(0))
        s_out = s_out / tmp
        output = self.manifold.k2p(s_out, self.c)
        return output

class HypMPGRU(nn.Module):
    def __init__(self, args, c):
        super(HypMPGRU, self).__init__()
        self.manifold = PoincareBall()
        self.c = c
        self.nhid = args.nhid
        self.W_ir = nn.Linear(self.nhid, self.nhid, bias=False)
        self.W_ih = nn.Linear(self.nhid, self.nhid, bias=False)
        self.W_iz = nn.Linear(self.nhid, self.nhid, bias=False)
        self.W_hr = nn.Linear(self.nhid, self.nhid, bias=False)
        self.W_hh = nn.Linear(self.nhid, self.nhid, bias=False)
        self.W_hz = nn.Linear(self.nhid, self.nhid, bias=False)
        if args.bias:
            self.bias = nn.Parameter(self.toHyperX(torch.ones(3, self.nhid) * 1e-5), requires_grad=True)
        else:
            self.register_buffer("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        geotorch.orthogonal(self.W_ir, "weight")
        geotorch.orthogonal(self.W_ih, "weight")
        geotorch.orthogonal(self.W_iz, "weight")
        geotorch.orthogonal(self.W_hr, "weight")
        geotorch.orthogonal(self.W_hh, "weight")
        geotorch.orthogonal(self.W_hz, "weight")

    def toHyperX(self, x, c=1.0):
        x_tan = self.manifold.proj_tan0(x, c)
        x_hyp = self.manifold.expmap0(x_tan, c)
        x_hyp = self.manifold.proj(x_hyp, c)
        return x_hyp

    def forward(self, hyperx, hyperh):
        out = self.mobius_gru_cell(hyperx, hyperh, self.bias)
        return out

    def mobius_gru_cell(self, input, hx, bias, nonlin=None):
        b_r, b_h, b_z = bias

        z_t = self.one_rnn_transform(self.W_hz, hx, self.W_iz, input, b_z).sigmoid()
        r_t = self.one_rnn_transform(self.W_hr, hx, self.W_ir, input, b_r).sigmoid()

        rh_t = r_t * hx

        h_tilde = torch.tanh(self.one_rnn_transform(self.W_hh, rh_t, self.W_ih, input, b_h)) # tanh

        hx = hx * z_t
        h_tilde = h_tilde * (1 - z_t)
        h_out = self.manifold.mobius_add(h_tilde, hx, c=self.c)
        return h_out

    def one_rnn_transform(self, W, h, U, x, b):
        W_otimes_h = W(h)
        U_otimes_x = U(x)
        Wh_plus_Ux = self.manifold.mobius_add(W_otimes_h, U_otimes_x, self.c)
        return self.manifold.mobius_add(Wh_plus_Ux, b, self.c)

    def mobius_linear(self, input, weight, bias=None, hyperbolic_input=True, hyperbolic_bias=True, nonlin=None):
        if hyperbolic_input:
            output = self.manifold.mobius_matvec(weight, input)
        else:
            output = torch.nn.functional.linear(input, weight)
            output = self.manifold.expmap0(output)
        if bias is not None:
            if not hyperbolic_bias:
                bias = self.manifold.expmap0(bias)
            output = self.manifold.mobius_add(output, bias)
        if nonlin is not None:
            output = self.manifold.mobius_fn_apply(nonlin, output)
        output = self.manifold.project(output)
        return output
