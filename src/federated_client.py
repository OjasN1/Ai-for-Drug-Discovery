import flwr as fl
import torch
import torch.nn as nn
from collections import OrderedDict
from src.model import MultiTaskModel


class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, device="cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def get_parameters(self, _config):
        return [v.cpu().numpy() for _, v in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        keys = list(self.model.state_dict().keys())
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, parameters)})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_y = nn.MSELoss()
        loss_t = nn.MSELoss()
        epochs = int(config.get("local_epochs", 1))

        for _ in range(epochs):
            for batch in self.train_loader:
                batch = batch.to(self.device)
                yp, tp = self.model(batch)
                y_true = batch.y[:,0] if batch.y.ndim > 1 else batch.y.view(-1)
                t_true = batch.tox[:,0] if hasattr(batch, "tox") else torch.zeros_like(y_true)
                loss = loss_y(yp.view(-1), y_true) + loss_t(tp.view(-1), t_true)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self.get_parameters({}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss = nn.MSELoss()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                yp, tp = self.model(batch)
                y_true = batch.y[:,0] if batch.y.ndim > 1 else batch.y.view(-1)
                l = loss(yp.view(-1), y_true)
                total_loss += l.item() * batch.num_graphs
                count += batch.num_graphs
        mse = total_loss / max(1, count)
        return mse, count, {"mse": mse}
