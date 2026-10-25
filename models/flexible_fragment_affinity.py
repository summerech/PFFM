import torch
import torch.nn as nn
from torch_scatter import scatter_max, scatter_mean, scatter_sum


class FragmentGraphLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_h, edge_index):
        if edge_index.numel() == 0:
            aggregate = torch.zeros_like(node_h)
        else:
            src, dst = edge_index
            message = self.message(torch.cat([node_h[src], node_h[dst]], dim=-1))
            aggregate = scatter_sum(message, dst, dim=0, dim_size=node_h.size(0))
            degree = scatter_sum(
                torch.ones_like(dst, dtype=node_h.dtype),
                dst,
                dim=0,
                dim_size=node_h.size(0),
            ).clamp(min=1.0).unsqueeze(-1)
            aggregate = aggregate / degree.sqrt()
        return self.norm(node_h + self.update(torch.cat([node_h, aggregate], dim=-1)))


class FlexibleFragmentAffinity(nn.Module):
    """Pose-light affinity ranker over a pocket point cloud and an rBRICS graph."""

    def __init__(self, vocab_size, hidden_dim=192, max_atomic_number=100, max_attachments=6):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_attachments = max_attachments
        self.protein_element = nn.Embedding(max_atomic_number + 1, hidden_dim)
        self.protein_geometry = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.protein_node = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)
        )
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.attachment_embedding = nn.Embedding(max_attachments + 1, hidden_dim)
        self.fragment_node = nn.Sequential(
            nn.Linear(2 * hidden_dim + 11, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)
        )
        self.fragment_layers = nn.ModuleList([FragmentGraphLayer(hidden_dim) for _ in range(3)])
        self.joint = nn.Sequential(
            nn.Linear(8 * hidden_dim + 2, 2 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.vina_head = nn.Linear(hidden_dim, 1)
        self.ha_head = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _pool(node_h, batch):
        mean = scatter_mean(node_h, batch, dim=0)
        maximum = scatter_max(node_h, batch, dim=0)[0]
        return mean, maximum

    def encode_protein(self, element, position, batch):
        center = scatter_mean(position, batch, dim=0)
        centered = position - center[batch]
        radius = centered.norm(dim=-1, keepdim=True)
        local_geometry = torch.zeros(
            (position.size(0), 3), dtype=position.dtype, device=position.device
        )
        for graph_index in range(int(batch.max().item()) + 1):
            index = torch.where(batch == graph_index)[0]
            distance = torch.cdist(position[index], position[index])
            distance.fill_diagonal_(float("inf"))
            nearest = distance.topk(
                min(4, max(distance.size(1) - 1, 1)), largest=False
            ).values
            local_geometry[index, 0] = nearest.mean(dim=-1) / 4.0
            local_geometry[index, 1] = (distance < 4.0).sum(dim=-1).float() / 32.0
            local_geometry[index, 2] = (distance < 6.0).sum(dim=-1).float() / 64.0
        geometry = torch.cat([radius / 12.0, local_geometry], dim=-1)
        node_h = self.protein_node(
            torch.cat(
                [
                    self.protein_element(element.clamp(min=0, max=100)),
                    self.protein_geometry(geometry),
                ],
                dim=-1,
            )
        )
        return self._pool(node_h, batch)

    def encode_fragments(
        self,
        token,
        attachment,
        size,
        batch,
        edge_index,
        fragment_feature,
        token_probability=None,
    ):
        token_h = (
            token_probability @ self.token_embedding.weight
            if token_probability is not None
            else self.token_embedding(token)
        )
        node_h = self.fragment_node(
            torch.cat(
                [
                    token_h,
                    self.attachment_embedding(attachment.clamp(max=self.max_attachments)),
                    size.float().unsqueeze(-1) / 16.0,
                    fragment_feature.float(),
                ],
                dim=-1,
            )
        )
        for layer in self.fragment_layers:
            node_h = layer(node_h, edge_index)
        return self._pool(node_h, batch)

    def forward(
        self,
        protein_element,
        protein_position,
        protein_batch,
        fragment_token,
        fragment_attachment,
        fragment_size,
        fragment_batch,
        fragment_edge_index,
        fragment_feature,
        token_probability=None,
    ):
        protein_mean, protein_max = self.encode_protein(
            protein_element, protein_position, protein_batch
        )
        fragment_mean, fragment_max = self.encode_fragments(
            fragment_token,
            fragment_attachment,
            fragment_size,
            fragment_batch,
            fragment_edge_index,
            fragment_feature,
            token_probability=token_probability,
        )
        fragment_count = scatter_sum(
            torch.ones_like(fragment_batch, dtype=fragment_mean.dtype),
            fragment_batch,
            dim=0,
        ).unsqueeze(-1)
        heavy_count = scatter_sum(
            fragment_size.float(), fragment_batch, dim=0
        ).unsqueeze(-1)
        joint = torch.cat(
            [
                protein_mean,
                protein_max,
                fragment_mean,
                fragment_max,
                protein_mean * fragment_mean,
                protein_max * fragment_max,
                (protein_mean - fragment_mean).abs(),
                (protein_max - fragment_max).abs(),
                fragment_count / 24.0,
                heavy_count / 64.0,
            ],
            dim=-1,
        )
        hidden = self.joint(joint)
        return self.vina_head(hidden).squeeze(-1), self.ha_head(hidden).squeeze(-1)


class FlexibleFragmentAffinityEnsemble(nn.Module):
    """Conservative differentiable ensemble in physical Vina units."""

    def __init__(self, members, means, stds, uncertainty_weight=0.35):
        super().__init__()
        if not members:
            raise ValueError("At least one affinity member is required.")
        self.members = nn.ModuleList(members)
        self.register_buffer("means", torch.as_tensor(means, dtype=torch.float32))
        self.register_buffer("stds", torch.as_tensor(stds, dtype=torch.float32))
        self.uncertainty_weight = float(uncertainty_weight)
        self.output_mean = float(self.means.mean())
        self.output_std = float(self.stds.mean().clamp(min=1e-4))

    def forward(self, *args, **kwargs):
        vina, ha = [], []
        for index, member in enumerate(self.members):
            normalized, logit = member(*args, **kwargs)
            vina.append(normalized * self.stds[index] + self.means[index])
            ha.append(logit)
        vina = torch.stack(vina, dim=0)
        conservative = vina.mean(dim=0) + self.uncertainty_weight * vina.std(
            dim=0, unbiased=False
        )
        normalized = (conservative - self.output_mean) / self.output_std
        return normalized, torch.stack(ha, dim=0).mean(dim=0)
