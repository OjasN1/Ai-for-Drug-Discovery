import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from src.data_processor import smiles_to_data
import torch

def graphs_from_csv(csv_path, smiles_col="smiles", target_col=None, default_target=0.0):
    df = pd.read_csv(csv_path)
    data_list = []
    for _, row in df.iterrows():
        data = smiles_to_data(str(row[smiles_col]))
        if data is None:
            continue
        if target_col is not None and target_col in row and pd.notna(row[target_col]):
            y = float(row[target_col])
        else:
            y = float(default_target)
        data.y = torch.tensor([y], dtype=torch.float)
        data_list.append(data)
    return data_list

def make_loaders(data_list, test_size=0.2, batch_size=32, seed=42):
    if len(data_list) == 0:
        raise RuntimeError("No valid graphs built")
    train_list, val_list = train_test_split(data_list, test_size=test_size, random_state=seed)
    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
