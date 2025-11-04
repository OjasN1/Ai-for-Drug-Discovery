import flwr as fl
import torch
import torch.nn as nn
from collections import OrderedDict
from src.model import MultiTaskModel


class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, device="cpu"):
        self.model = model.to(device)  # <-- MOVES MODEL TO GPU/MPS
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device  # <-- STORES THE DEVICE (e.g., "mps")

    def get_parameters(self, config):
        """Returns the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Changes the parameters of the model using the given ones."""
        keys = [key for key in self.model.state_dict().keys()]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Implements distributed training for a given client."""
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_y = nn.MSELoss()
        loss_t = nn.MSELoss()
        epochs = int(config.get("local_epochs", 1))

        for _ in range(epochs):
            for batch in self.train_loader:
                batch = batch.to(self.device)  # <-- MOVES BATCH DATA TO GPU
                yp, tp = self.model(batch)

                y_true = batch.y.view(-1)
                t_true = batch.tox.view(-1) if hasattr(batch, "tox") else torch.zeros_like(y_true)

                loss = loss_y(yp.view(-1), y_true) + loss_t(tp.view(-1), t_true)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self.get_parameters({}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluates the model on the local validation set."""
        if not self.val_loader:
            return float(0.0), 0, {"mse": float(0.0)}

        self.set_parameters(parameters)
        self.model.eval()
        loss_y = nn.MSELoss()
        loss_t = nn.MSELoss()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)  # <-- MOVES BATCH DATA TO GPU
                yp, tp = self.model(batch)

                y_true = batch.y.view(-1)
                t_true = batch.tox.view(-1) if hasattr(batch, "tox") else torch.zeros_like(y_true)

                loss = loss_y(yp.view(-1), y_true) + loss_t(tp.view(-1), t_true)
                total_loss += loss.item() * batch.num_graphs
                count += batch.num_graphs

        mse = total_loss / max(1, count)
        return float(mse), count, {"mse": float(mse)}