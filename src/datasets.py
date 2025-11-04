import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from src.data_processor import smiles_to_data
import torch


def graphs_from_csv(csv_file, smiles_col, target_col, default_target=0.0):
    df = pd.read_csv(csv_file)
    data_list = []

    for idx, row in df.iterrows():
        smiles = row[smiles_col]
        try:
            target = float(row[target_col])
        except (ValueError, TypeError):
            target = float(default_target)

        data = smiles_to_data(smiles)
        if data:
            data.y = torch.tensor([target], dtype=torch.float)
            data.smiles = smiles  # Attach SMILES for later reference
            data_list.append(data)

    return data_list


# --- NEW: Added num_workers parameter ---
def make_loaders(data_list, test_size=0.2, batch_size=32, num_workers=0):
    if not data_list:
        return None, None

    train_data, val_data = train_test_split(data_list, test_size=test_size, random_state=42)

    # Ensure we don't request more workers than samples
    train_workers = min(num_workers, len(train_data)) if len(train_data) > 0 else 0
    val_workers = min(num_workers, len(val_data)) if len(val_data) > 0 else 0

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_workers,
        pin_memory=True
    )

    val_loader = None
    if val_data:
        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=val_workers,
            pin_memory=True
        )

    return train_loader, val_loader