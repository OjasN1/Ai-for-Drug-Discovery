import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from src.model import MultiTaskModel
from src.data_processor import smiles_to_data
import numpy as np

class LocalTrainer:
    def __init__(self, in_dim, hidden=128, device="cpu"):
        self.device = device
        self.model = MultiTaskModel(in_dim=in_dim, hidden=hidden).to(device)
        self.loss_y = nn.MSELoss()
        self.loss_t = nn.MSELoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def fit(self, train_loader, val_loader=None, epochs=10):
        for ep in range(epochs):
            self.model.train()
            running_loss = 0.0
            for batch in train_loader:
                batch = batch.to(self.device)
                yp, tp = self.model(batch)
                y_true = batch.y[:,0] if batch.y.ndim > 1 else batch.y.view(-1)
                if hasattr(batch, "tox"):
                    if batch.tox.ndim > 1:
                        t_true = batch.tox[:,0]
                    else:
                        t_true = batch.tox
                else:
                    t_true = torch.zeros_like(y_true)
                loss = self.loss_y(yp.view(-1), y_true) + self.loss_t(tp.view(-1), t_true)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                running_loss += loss.item() * batch.num_graphs
            avg_loss = running_loss / len(train_loader.dataset)
            print("Epoch {}/{} train loss: {:.4f}".format(ep+1, epochs, avg_loss))
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self.device)
                        yp, tp = self.model(batch)
                        y_true = batch.y[:,0] if batch.y.ndim > 1 else batch.y.view(-1)
                        if hasattr(batch, "tox"):
                            if batch.tox.ndim > 1:
                                t_true = batch.tox[:,0]
                            else:
                                t_true = batch.tox
                        else:
                            t_true = torch.zeros_like(y_true)
                        loss = self.loss_y(yp.view(-1), y_true) + self.loss_t(tp.view(-1), t_true)
                        val_loss += loss.item() * batch.num_graphs
                val_loss /= len(val_loader.dataset)
                print("Epoch {}/{} val loss: {:.4f}".format(ep+1, epochs, val_loss))

    def predict_yield(self, smiles_list):
        self.model.eval()
        preds = []
        with torch.no_grad():
            for s in smiles_list:
                d = smiles_to_data(s)
                if d is None:
                    preds.append(0.0)
                    continue
                d.batch = torch.zeros(d.x.size(0), dtype=torch.long)
                d = d.to(self.device)
                yp, _ = self.model(d)
                preds.append(float(yp.view(-1).cpu().item()))
        return np.array(preds)

    def predict_tox(self, smiles_list):
        self.model.eval()
        preds = []
        with torch.no_grad():
            for s in smiles_list:
                d = smiles_to_data(s)
                if d is None:
                    preds.append(1.0)  # Assume high toxicity for invalid smiles
                    continue
                d.batch = torch.zeros(d.x.size(0), dtype=torch.long)
                d = d.to(self.device)
                _, tp = self.model(d)
                preds.append(float(tp.view(-1).cpu().item()))
        return np.array(preds)
