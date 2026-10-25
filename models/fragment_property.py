import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_sum


class FragmentPropertyPredictor(nn.Module):
    """Condition-free molecular property predictor over rBRICS fragment graphs."""

    def __init__(
        self,
        vocab_size,
        hidden_dim=192,
        max_attachments=6,
        max_fragments=24,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_attachments = max_attachments
        self.max_fragments = max_fragments
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.attachment_embedding = nn.Embedding(max_attachments + 1, hidden_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.graph_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _predict(self, token_embedding, attachment_embedding, fragment_size, fragment_batch):
        node_h = self.node_mlp(
            torch.cat([token_embedding, attachment_embedding, fragment_size], dim=-1)
        )
        graph_mean = scatter_mean(node_h, fragment_batch, dim=0)
        graph_sum = scatter_sum(node_h, fragment_batch, dim=0)
        count = scatter_sum(
            torch.ones_like(fragment_batch, dtype=node_h.dtype), fragment_batch, dim=0
        ).unsqueeze(-1)
        graph_sum = graph_sum / count.sqrt().clamp(min=1.0)
        graph_h = torch.cat(
            [graph_mean, graph_sum, count / self.max_fragments], dim=-1
        )
        return self.graph_mlp(graph_h).squeeze(-1).sigmoid()

    def forward(
        self,
        fragment_token,
        attachment_count,
        fragment_batch,
        token_size_lookup,
    ):
        attachment_count = attachment_count.clamp(max=self.max_attachments)
        fragment_size = token_size_lookup[fragment_token].float().unsqueeze(-1)
        fragment_size = fragment_size / 32.0
        return self._predict(
            self.token_embedding(fragment_token),
            self.attachment_embedding(attachment_count),
            fragment_size,
            fragment_batch,
        )

    def forward_soft(
        self,
        token_probability,
        attachment_probability,
        fragment_batch,
        token_size_lookup,
    ):
        token_embedding = token_probability @ self.token_embedding.weight
        attachment_embedding = attachment_probability @ self.attachment_embedding.weight
        fragment_size = (
            token_probability @ token_size_lookup.to(token_probability)
        ).unsqueeze(-1) / 32.0
        return self._predict(
            token_embedding,
            attachment_embedding,
            fragment_size,
            fragment_batch,
        )
