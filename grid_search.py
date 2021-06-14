import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import itertools
import json


def make_cases():
    model_cases = ["lstm", "gru"]
    stack_cases = [1, 2]
    scaler_cases = ["minmax", "std"]
    all_case = list(itertools.product(*[model_cases, stack_cases, scaler_cases]))
    return all_case


def set_seed():
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def data_loader(scaler):
    csv_data = pd.read_csv("data/{}.csv".format(scaler))
    x = csv_data[["Close", "Open", "High", "Low", "Volume", "Change"]].to_numpy()
    y = csv_data[["Low"]].to_numpy()

    dataX = []
    dataY = []

    for i in range(0, len(y) - 31):
        _x = x[i:i + 30]
        _y = y[i + 31]
        dataX.append(_x)
        dataY.append(_y)

    trainX = dataX[:int(len(dataX) * 0.8)]
    trainY = dataY[:int(len(dataX) * 0.8)]
    evalX = dataX[int(len(dataX) * 0.8):]
    evalY = dataY[int(len(dataX) * 0.8):]

    trainX = torch.from_numpy(np.array(trainX))
    trainY = torch.from_numpy(np.array(trainY))
    evalX = torch.from_numpy(np.array(evalX))
    evalY = torch.from_numpy(np.array(evalY))

    return trainX, trainY, evalX, evalY


def get_validation_loss(model, trainX, trainY, evalX, evalY):
    result = dict()
    model = model
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(1, 1001):
        pred = model(trainX.float().cuda())
        loss = criterion(pred, trainY.float().cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
          with torch.no_grad():
              valid_predict = model(evalX.float().cuda())
              eval_loss = criterion(valid_predict, evalY.float().cuda())
              result[epoch] = eval_loss.item()/len(evalY)
    return result


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out


def main():
    all_loss = dict()
    set_seed()
    all_cases = make_cases()
    for model_case, stack, scaler in all_cases:
        trainX, trainY, evalX, evalY = data_loader(scaler)
        if model_case == "lstm":
            model = LSTM(input_dim=6, hidden_dim=6, output_dim=1, num_layers=stack).cuda()
            result = get_validation_loss(model, trainX, trainY, evalX, evalY)
            for epoch, loss in result.items():
                name = "{}_{}_{}_{}".format(model_case, stack, scaler, epoch)
                all_loss[name] = loss
        elif model_case == "gru":
            model = GRU(input_dim=6, hidden_dim=6, output_dim=1, num_layers=stack).cuda()
            result = get_validation_loss(model, trainX, trainY, evalX, evalY)
            for epoch, loss in result.items():
                name = "{}_{}_{}_{}".format(model_case, stack, scaler, epoch)
                all_loss[name] = loss
    with open('parmas/result.json', 'w') as f:
        json.dump(all_loss, f, indent="\t")
    f.close()