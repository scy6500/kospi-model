import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime


def data_loader(scaler):
    csv_data = pd.read_csv("./data/{}.csv".format(scaler))
    x = csv_data[["Close", "Open", "High", "Low", "Volume", "Change"]].to_numpy()
    y = csv_data[["Low"]].to_numpy()

    dataX = []
    dataY = []

    for i in range(0, len(y) - 31):
        _x = x[i:i + 30]
        _y = y[i + 31]
        dataX.append(_x)
        dataY.append(_y)

    dataX = torch.from_numpy(np.array(dataX))
    dataY = torch.from_numpy(np.array(dataY))

    return dataX, dataY


def train(model, dataX, dataY, epochs):
    model = model
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters())
    model.train()
    for epoch in range(epochs):
        pred = model(dataX.float().cuda())
        loss = criterion(pred, dataY.float().cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), "./model/{}}.pt".format(datetime.today().date()))


def save_modelinfo(model, stack, scaler, epoch):
    model_info = dict()
    model_info["model"] = model
    model_info["stack"] = stack
    model_info["scaler"] = scaler
    model_info["epoch"] = epoch
    # if scaler == "minmax":
    #     with open('data/minmax.json', 'r') as f:
    #         minmax_data = json.load(f)
    #         model_info["Low_max"] = minmax_data["Low_max"]
    #         model_info["Low_min"] = minmax_data["Low_min"]
    #     f.close()
    # elif scaler == "std":
    #     with open('data/std.json', 'r') as f:
    #         std_data = json.load(f)
    #         model_info["Low_average"] = std_data["Low_average"]
    #         model_info["Low_std"] = std_data["Low_std"]
    #     f.close()
    with open('model/model_info.json', 'w') as f:
        json.dump(model_info, f, indent="\t")
    f.close()

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
    with open('./params/result.json', 'r') as f:
        gridsearch_result = json.load(f)
    f.close()
    best_param = sorted(gridsearch_result.items(), key=lambda x: x[1])[0][0].split("_")
    model, stack, scaler, epoch = best_param[0], int(best_param[1]), best_param[2], int(best_param[3])
    dataX, dataY = data_loader(scaler)
    if model == "lstm":
        model = LSTM(input_dim=6, hidden_dim=6, output_dim=1, num_layers=stack).cuda()
        train(model, dataX, dataY, epoch)
    elif model == "gru":
        model = GRU(input_dim=6, hidden_dim=6, output_dim=1, num_layers=stack).cuda()
        train(model, dataX, dataY, epoch)