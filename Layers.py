from torch import nn
import torch
from torch.autograd import Function, Variable
import torch.nn.functional as Func
from torch.nn import Module, Parameter
import math
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")


def matrix_fnorm(W):
    # W:(h,n,n) return (h)
    h, n, n = W.shape
    W = W ** 2
    norm = (W.sum(dim=1).sum(dim=1)) ** (0.5)
    return norm / (n ** 0.5)


class MarketGRU_gate(nn.Module):
    def __init__(self, input_size, output_size, activation, stocks):
        super(MarketGRU_gate, self).__init__()
        self.activation = activation
        self.W = Parameter(torch.FloatTensor(stocks, input_size, output_size))
        self.bias = Parameter(torch.zeros(stocks, output_size))
        self.reset_param(self.W)

    def reset_param(self, t):
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return self.activation(torch.matmul(x, self.W).squeeze() + self.bias)


class MarketGRUcell(nn.Module):
    def __init__(self, in_feat, in_market, out_feat, stocks):
        super(MarketGRUcell, self).__init__()
        self.in_feat = in_feat
        self.in_market = in_market
        self.out_feat = out_feat

        self.marketgate = MarketGRU_gate(in_feat + in_market, in_market, nn.Sigmoid(), stocks)
        self.merge = MarketGRU_gate(in_feat + in_market, out_feat, nn.Tanh(), stocks)

        self.update = MarketGRU_gate(2 * out_feat, out_feat, nn.Sigmoid(), stocks)
        self.reset = MarketGRU_gate(2 * out_feat, out_feat, nn.Sigmoid(), stocks)
        self.hat = MarketGRU_gate(2 * out_feat, out_feat, nn.Tanh(), stocks)

    #
    def forward(self, xt, markett, hidden):
        # print(torch.unsqueeze(torch.cat([xt,markett.expand((xt.shape[0],markett.shape[-1]))],dim=-1),dim=1).shape)
        #                                               N,d1+d2
        marketgate = self.marketgate(
            torch.unsqueeze(torch.cat([xt, markett.expand((xt.shape[0], markett.shape[-1]))], dim=-1), dim=1))
        hxm = self.merge(
            torch.unsqueeze(torch.cat([xt, marketgate * markett.expand(xt.shape[0], markett.shape[-1])], dim=-1),
                            dim=1))

        zt = self.update(torch.unsqueeze(torch.cat((hidden, hxm), dim=-1), dim=1))
        rt = self.reset(torch.unsqueeze(torch.cat((hidden, hxm), dim=-1), dim=1))
        h_t = self.hat(torch.unsqueeze(torch.cat((hidden * rt, hxm), dim=-1), dim=1))
        ht = (1 - zt) * hidden + zt * h_t

        return ht


class MarketGRU(nn.Module):
    def __init__(self, in_feat, in_market, hidden_feat, time_length, stocks) -> None:
        super(MarketGRU, self).__init__()
        self.in_feat = in_feat
        self.in_market = in_market
        self.hidden_feat = hidden_feat
        self.time_length = time_length
        self.stocks = stocks
        self.marketgru_cell = MarketGRUcell(self.in_feat, self.in_market, self.hidden_feat, stocks)

    def forward(self, x, market, hidden=None):
        h = []
        if hidden == None:
            hidden = torch.zeros((self.stocks, self.hidden_feat), device=x.device, dtype=x.dtype)

        for t in range(self.time_length):
            hidden = self.marketgru_cell(x[:, t, :], market[t], hidden)
            h.append(hidden)
        h = torch.stack(h, dim=1)
        return h, hidden


class RelationMapping(nn.Module):
    def __init__(self, infeat) -> None:
        super(RelationMapping, self).__init__()
        self.fc1 = nn.Linear(infeat, int(infeat / 2))
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(int(infeat / 2), 1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        out = self.relu2(x)
        return out


class MarketRelation(nn.Module):
    def __init__(self, infeat, time_length, alpha=0, beta=.1) -> None:
        super(MarketRelation, self).__init__()
        self.mapping = RelationMapping(infeat + infeat)
        self.time_length = time_length
        self.alpha = alpha
        self.beta = beta
        self.a = nn.Parameter(torch.randn(self.time_length, 1))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, x):
        Adj = []
        n, d = x[:, 0, :].shape[0], x[:, 0, :].shape[1]
        for i in range(self.time_length):
            x_expanded = x[:, i, :].unsqueeze(1).expand(n, n, d)
            x_expanded_transposed = x_expanded.transpose(0, 1)
            out = torch.cat((x_expanded, x_expanded_transposed), dim=2)
            adj_i = self.mapping(out).squeeze()
            Adj.append(adj_i)

        Adj = torch.stack(Adj, dim=0)  # T*N*N
        a = torch.softmax(self.a, dim=0)
        A = []
        for i in range(self.time_length):
            A.append(Adj[i] * a[i] * (1 + self.alpha * math.exp(-self.beta * (self.time_length - i - 1))))
        A = torch.stack(A)
        A = torch.sum(A, dim=0)
        return A


class RelationSelection(nn.Module):
    def __init__(self, infeat, externK) -> None:
        super(RelationSelection, self).__init__()
        self.fc1 = nn.Linear(infeat, int(infeat / 2))
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(int(infeat / 2), externK)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        out = self.relu2(x)
        return out


class NobsRelation(nn.Module):
    def __init__(self, infeat, stocks, externK, externC, device='cuda:0'):
        super(NobsRelation, self).__init__()
        self.stocks = stocks
        self.device = device
        self.externK = externK
        self.select = RelationSelection(infeat * 2, externK)
        self.mapping = RelationMapping(externC * 2)
        self.E = nn.Parameter(torch.empty(size=(externK, stocks, externC)))
        nn.init.xavier_uniform_(self.E.data, gain=1.414)

    def forward(self, x):
        D = []
        n, d = x.shape
        x_expanded = x.unsqueeze(1).expand(n, n, d)
        x_expanded_transposed = x_expanded.transpose(0, 1)
        out = torch.cat((x_expanded, x_expanded_transposed), dim=2)
        a = self.select(out)
        a = F.gumbel_softmax(a, dim=-1, hard=False)

        for i in range(self.externK):
            D.append(torch.matmul(self.E[i], self.E[i].T))

        D = torch.stack(D, dim=2)
        A = torch.sum(D * a, dim=2)
        return A


class GetRelation(nn.Module):
    def __init__(self, infeat, time_length, stocks, externK, externC, device='cuda:0', isNorm=False):
        super(GetRelation, self).__init__()
        self.isNorm = isNorm
        self.mrelation = MarketRelation(infeat, time_length)
        self.nobsrelation = NobsRelation(infeat, stocks, externK, externC, device)

    def forward(self, x, Ind, Loc):
        A = []
        if self.isNorm:
            A.append(self.laplacian(Ind))
            A.append(self.laplacian(Loc))
            A.append(self.laplacian(self.DynamicAjd(x)))
            A.append(self.laplacian(self.ExternAdj()))
        else:
            A.append(Ind)
            A.append(Loc)
            A.append(self.mrelation(x))
            A.append(self.nobsrelation(x[:, -1, :]))
            A = torch.stack(A)
        return A

    def laplacian(self, W):
        N, N = W.shape
        W = W + torch.eye(N).to(W.device)
        D = W.sum(axis=1)
        D = torch.diag(D ** (-0.5))
        out = D @ W @ D
        return out


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = input @ self.weight
        output = adj @ support
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class RelationAggregation(nn.Module):
    def __init__(self, hidden_feat, relationtypes):
        super(RelationAggregation, self).__init__()
        self.gnns = [GraphConvolution(hidden_feat, hidden_feat) for _ in
                     range(relationtypes)]
        for i, gnn in enumerate(self.gnns):
            self.add_module('attention_{}'.format(i), gnn)
        self.out = nn.Linear(hidden_feat * relationtypes, hidden_feat)

    def forward(self, x, adj):
        h = torch.zeros_like(x).to(x.device)
        for i, gnn in enumerate(self.gnns):
            h += gnn(x, adj[i])
        return h


class AttributrMultiplexNetwork(nn.Module):
    def __init__(self, infeat, time_length, stocks, hidden_feat, relationtypes, externK, externC) -> None:
        super(AttributrMultiplexNetwork, self).__init__()
        self.in_feat = hidden_feat
        self.out_feat = hidden_feat
        self.relations = relationtypes
        self.stocks = stocks
        self.aggregation = RelationAggregation(hidden_feat, relationtypes)

        # self.gru=nn.GRU(5,64,batch_first=True)
        self.attentionscore = nn.Parameter(torch.randn((self.relations, self.stocks, self.stocks)))
        nn.init.xavier_uniform_(self.attentionscore.data, gain=1.414)
        self.weight = nn.Parameter(torch.randn(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        self.getrelation = GetRelation(hidden_feat, time_length, stocks, externK, externC)

    def forward(self, x, Ind, Loc):
        # [C,N,N]
        A = self.getrelation(x, Ind, Loc)
        A = torch.mul(torch.softmax(self.attentionscore, dim=0), A)
        out = self.aggregation(x[:, -1, :], A)

        return out
