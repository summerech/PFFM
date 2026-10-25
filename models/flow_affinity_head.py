import torch
import torch.nn as nn


class FlowAffinityHead(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim), nn.SiLU(), nn.Dropout(0.15),
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.vina_head = nn.Linear(hidden_dim, 1)
        self.ha_head = nn.Linear(hidden_dim, 1)

    def forward(self, fragment_graph_h, protein_graph_h):
        hidden = self.network(torch.cat([fragment_graph_h, protein_graph_h], dim=-1))
        return self.vina_head(hidden).squeeze(-1), self.ha_head(hidden).squeeze(-1)
