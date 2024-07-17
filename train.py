import os
import argparse
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, matthews_corrcoef
from torch import nn, optim
import numpy as np
import os
import time
import scipy.sparse as sp
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from Model import *
import copy

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch import nn, optim

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:0',
                    help='GPU to use')

parser.add_argument('--market', type=str, default='CSI500',
                    help='GPU to use')

parser.add_argument('--model', type=str, default='(HPKE-GNN', )

parser.add_argument('--epochs', type=int, default=100,
                    help='Training max epoch')

parser.add_argument('--lr', type=float, default=5e-4,
                    help='Learning rate')

parser.add_argument('--in_feat', type=int, default=9,
                    help='five attributes:open,close,high,low,vol')

parser.add_argument('--in_market', type=int, default=5,
                    help='five attributes:open,close,high,low,vol')

parser.add_argument('--clas', type=int, default=3,
                    help='3 class for classification, increase or decrease')

parser.add_argument('--weight_constraint', type=float, default=0.,
                    help='L2 weight constraint')

parser.add_argument('--time_length', type=int, default=20,
                    help='rnn length')

parser.add_argument('--hidden_feat', type=int, default=32,
                    help='dropout rate')

parser.add_argument('--cat_add', type=str, default='cat',
                    help='cat or add')

parser.add_argument('--batch_size', type=int, default=16,
                    help='dropout rate')

parser.add_argument('--mode', type=str, default='classification',
                    help='regression or classification')

parser.add_argument('--relationtypes', type=int, default=4,
                    help='dropout rate')

parser.add_argument('--externK', type=int, default=10,
                    help='dropout rate')

parser.add_argument('--externC', type=int, default=32,
                    help='dropout rate')


def PreProcess(data, Market, args):
    print('Loading data...')
    data = data[:, 5:]
    Market = Market[5:]
    close_price = data[:, :, 1].copy()  # Close Data
    label = np.zeros_like(close_price)
    normed_data = np.zeros_like(data)

    for i, x in enumerate(data):
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        normed_data[i] = (x - mean) / std

    if args.mode == 'classification':
        for i, _ in enumerate(close_price):  # Trend
            for j in range(1, _.shape[0]):
                if (close_price[i, j] - close_price[i, j - 1]) / close_price[i, j - 1] * 100 > .5:
                    label[i, j] = 2
                elif (close_price[i, j] - close_price[i, j - 1]) / close_price[i, j - 1] * 100 < -.5:
                    label[i, j] = 0
                else:
                    label[i, j] = 1

    m = np.mean(Market, axis=0)
    s = np.std(Market, axis=0)
    Market = (Market - m) / s

    print('Done!')
    return normed_data, label, Market


def test(model, eval_data, eval_label, eval_market, Ind, Loc):
    model.eval()
    seq_len = eval_data.size(1)
    seq = list(range(seq_len))[time_length:]
    preds = []
    trues = []
    for day in seq:
        out = model(eval_data[:, day - time_length: day, :], eval_market[day - time_length: day, :], Ind, Loc)
        output = torch.argmax(out, dim=-1)
        preds.append(output.cpu().numpy())
        trues.append(eval_label[:, day].cpu().numpy())
    acc = accuracy_score(torch.tensor(trues).reshape(-1), torch.tensor(preds).reshape(-1))
    macc = matthews_corrcoef(torch.tensor(trues).reshape(-1), torch.tensor(preds).reshape(-1))

    return acc, macc


def train(args, model, bestmodel, train_data, train_label, train_market,
          eval_data, eval_label, eval_market,
          Ind, Loc, ):
    bestacc = 0
    seq_len = train_data.size(1)
    for epoch in range(args.epochs):
        model.train()
        count_train = 0
        print('epoch', epoch + 1, ":")
        train_seq = list(range(seq_len))[time_length:]
        random.shuffle(train_seq)
        avg_loss = 0
        for day in train_seq:
            out = model(train_data[:, day - time_length:day, :], train_market[day - time_length:day], Ind, Loc)
            loss = criterion(out, train_label[:, day])
            avg_loss += loss
            loss.backward()
            if (count_train % args.batch_size) == 0:
                optimizer.step()
                optimizer.zero_grad()
        if (count_train % args.batch_size) != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = avg_loss / len(train_seq)

        acc, mcc = test(model, eval_data, eval_label, eval_market, Ind, Loc)
        print(
            'train_loss={:.5f},eval_acc={:.2f},eval_mcc={:.5f}'.format(avg_loss.item(), acc * 100, mcc))
        if acc > bestacc:
            bestmodel = copy.deepcopy(model)
            bestacc = acc

    return bestmodel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    # set_seed(2024)
    for z in range(3):
        args = parser.parse_args()
        DEVICE = args.device

        Daily = np.load('datasets/daily_indicator.npy')
        Weekly = np.load('datasets/weekly_indicator.npy')[:, :, [0, 2, 3, 4]]
        MarketIdx = np.load('datasets/market_indicator.npy')
        Ind = torch.from_numpy(np.load('datasets/relations/Industry.npy')).float().to(DEVICE)
        Loc = torch.from_numpy(np.load('datasets/relations/Location.npy')).float().to(DEVICE)
        IndicatorData = np.concatenate([Daily, Weekly], axis=-1)

        data, label, MarketIdx = PreProcess(IndicatorData, MarketIdx, args)
        data = torch.from_numpy(data).float().to(DEVICE)
        label = torch.from_numpy(label).long().to(DEVICE)
        MarketIdx = torch.from_numpy(MarketIdx).float().to(DEVICE)

        stocks = data.size(0)
        days = data.size(1)
        feats = data.size(2)

        time_length = args.time_length
        test_days = 150
        train_indicator = data[:, :-test_days]
        eval_indicator = data[:, -int(test_days / 2) - time_length:]
        test_indicator = data[:, -test_days - time_length: -int(test_days / 2)]

        train_label = label[:, : -test_days]
        eval_label = label[:, -int(test_days / 2) - time_length:]
        test_label = label[:, -test_days - time_length: -int(test_days / 2)]

        train_marketidx = MarketIdx[:-test_days]
        eval_marketidx = MarketIdx[-int(test_days / 2) - time_length:]
        test_marketidx = MarketIdx[-test_days - time_length: -int(test_days / 2)]

        criterion = torch.nn.CrossEntropyLoss()

        model = Model(in_feat=args.in_feat, in_market=args.in_market, hidden_feat=args.hidden_feat,
                      time_length=time_length, stocks=stocks, clas=args.clas, relationtypes=args.relationtypes,
                      externK=20, externC=args.externC)
        bestmodel = None
        model.cuda(device=DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_constraint)

        bestmodel = train(args, model, bestmodel, train_indicator, train_label, train_marketidx,
                          eval_indicator, eval_label, eval_marketidx,
                          Ind, Loc, )

        test_acc, test_mcc = test(bestmodel, test_indicator, test_label, test_marketidx, Ind, Loc)

        # torch.save(bestmodel.state_dict(), 'model.pth')
        print('finally: acc:{:.2f},mcc:{:.4f}\n'.format(test_acc * 100, test_mcc))

