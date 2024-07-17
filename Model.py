from torch import nn
import torch
from torch.autograd import Function, Variable
import torch.nn.functional as Func
from torch.nn import Module, Parameter
import math
import torch.nn.functional as F
import warnings
from Layers import *

class Model(nn.Module):
    def __init__(self, in_feat, in_market, hidden_feat, time_length, stocks, clas, relationtypes, externK,
                 externC):
        super(Model, self).__init__()
        self.stocks = stocks
        self.MarketGRU = MarketGRU(in_feat, in_market, hidden_feat, time_length, stocks)

        self.Relational = AttributrMultiplexNetwork(in_feat, time_length, stocks, hidden_feat, relationtypes, externK,
                                              externC)
        self.predict = nn.Linear(in_feat + hidden_feat * 2, clas)
        self.drop = nn.Dropout(p=.5)

    def forward(self, x, market, Ind, Loc):
        h, hidden = self.MarketGRU(x, market)
        h = self.drop(h)
        RelationEmbedding = self.Relational(h, Ind, Loc)
        RelationEmbedding = self.drop(RelationEmbedding)
        out = self.predict(torch.cat([x[:, -1, :], hidden, RelationEmbedding], dim=-1))
        return out


if __name__ == '__main__':
    x = torch.ones(500, 64)
    model = NobsRelation(64, 500, 8, 32)
    h = model(x)
    print(h.shape)
