import torch
import torch.nn as nn
import pandas as pd  # <-- THIS IS THE FIX
from src.model import MultiTaskModel
from src.datasets import graphs_from_csv
from torch_geometric.loader import DataLoader
from tqdm import tqdm


class LocalTrainer:
    # --- Added num_workers to __init__ ---
    def __init__(self, in_dim, hidden, device="cpu", num_workers=0):
        self.model = MultiTaskModel(in_dim=in_dim, hidden=hidden).to(device)
        self.device = device
        self.num_workers = num_workers  # Store num_workers
        self.y_criterion = nn.MSELoss()
        self.t_criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def fit(self, train_loader, val_loader, epochs):
        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} (Train)", leave=False):
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                yp, tp = self.model(batch)

                y_true = batch.y.view(-1)
                t_true = batch.tox.view(-1) if hasattr(batch, "tox") else torch.zeros_like(y_true)

                loss = self.y_criterion(yp.view(-1), y_true) + self.t_criterion(tp.view(-1), t_true)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * batch.num_graphs

            train_loss /= len(train_loader.dataset)

            val_loss = self.evaluate(val_loader)
            print(f"Epoch {epoch}/{epochs} train loss: {train_loss:.4f}")
            print(f"Epoch {epoch}/{epochs} val loss: {val_loss:.4f}")

    def evaluate(self, loader):
        if loader is None:
            return 0.0
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc="Validating", leave=False):
                batch = batch.to(self.device)
                yp, tp = self.model(batch)

                y_true = batch.y.view(-1)
                t_true = batch.tox.view(-1) if hasattr(batch, "tox") else torch.zeros_like(y_true)

                loss = self.y_criterion(yp.view(-1), y_true) + self.t_criterion(tp.view(-1), t_true)
                total_loss += loss.item() * batch.num_graphs
        return total_loss / len(loader.dataset)

    def predict_yield(self, smiles_list):
        self.model.eval()
        data_list = graphs_from_csv(pd.DataFrame({"smiles": smiles_list, "yield": [0] * len(smiles_list)}), "smiles",
                                    "yield")
        if not data_list:
            return [0] * len(smiles_list)

        # --- Using num_workers for faster inference ---
        loader = DataLoader(
            data_list,
            batch_size=32,
            shuffle=False,
            num_workers=self.num_workers
        )

        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                yp, _ = self.model(batch)
                preds.extend(yp.view(-1).tolist())
        return preds

    def predict_tox(self, smiles_list):
        self.model.eval()
        data_list = graphs_from_csv(pd.DataFrame({"smiles": smiles_list, "yield": [0] * len(smiles_list)}), "smiles",
                                    "yield")
        if not data_list:
            return [0] * len(smiles_list)

        # --- Using num_workers for faster inference ---
        loader = DataLoader(
            data_list,
            batch_size=32,
            shuffle=False,
            num_workers=self.num_workers
        )

        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                _, tp = self.model(batch)
                preds.extend(tp.view(-1).tolist())
        return preds