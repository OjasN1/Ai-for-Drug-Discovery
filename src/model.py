import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=3):
        super(GraphEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden))
        for _ in range(layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.dropout = nn.Dropout(0.2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))
            x = self.dropout(x)
        g = global_mean_pool(x, data.batch)  # Aggregate node features to graph feature
        return g

class YieldHead(nn.Module):
    def __init__(self, hidden=128):
        super(YieldHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1)
        )
    def forward(self, graph_feat):
        return self.mlp(graph_feat)

class ToxicityHead(nn.Module):
    def __init__(self, hidden=128):
        super(ToxicityHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1)
        )
    def forward(self, graph_feat):
        return self.mlp(graph_feat)

class MultiTaskModel(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super(MultiTaskModel, self).__init__()
        self.encoder = GraphEncoder(in_dim, hidden)
        self.yield_head = YieldHead(hidden)
        self.tox_head = ToxicityHead(hidden)

    def forward(self, data):
        graph_feat = self.encoder(data)
        yield_pred = self.yield_head(graph_feat)
        tox_pred = self.tox_head(graph_feat)
        return yield_pred, tox_pred
