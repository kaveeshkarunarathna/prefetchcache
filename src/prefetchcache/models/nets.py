"""LSTM and small Transformer next-request predictors.

Both take the same input: the last ``max_len`` endpoint ids of a session,
left-padded with a PAD id, shape [batch, max_len]. Both return logits over
the next endpoint at every position, shape [batch, max_len, n_endpoints].
The serving path reads the last position only.
"""
from __future__ import annotations

import torch
from torch import nn


class LSTMNet(nn.Module):
    def __init__(self, n_endpoints: int, max_len: int = 20, emb: int = 64, hidden: int = 128, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.pad_id = n_endpoints
        self.embed = nn.Embedding(n_endpoints + 1, emb)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, n_endpoints)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(self.drop(self.embed(x)))
        return self.out(self.drop(h))


class TransformerNet(nn.Module):
    """Causal Transformer encoder with learned positions (SASRec-style)."""

    def __init__(self, n_endpoints: int, max_len: int = 20, emb: int = 64, heads: int = 4, layers: int = 2, ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.pad_id = n_endpoints
        self.embed = nn.Embedding(n_endpoints + 1, emb)
        self.pos = nn.Embedding(max_len, emb)
        self.drop = nn.Dropout(dropout)
        block = nn.TransformerEncoderLayer(emb, heads, ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(emb)
        self.out = nn.Linear(emb, n_endpoints)
        self.register_buffer("positions", torch.arange(max_len).unsqueeze(0), persistent=False)
        self.register_buffer("causal", torch.triu(torch.full((max_len, max_len), float("-inf")), diagonal=1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.embed(x) + self.pos(self.positions))
        h = self.encoder(h, mask=self.causal)
        return self.out(self.norm(h))


def build_model(kind: str, n_endpoints: int, max_len: int = 20) -> nn.Module:
    if kind == "lstm":
        return LSTMNet(n_endpoints, max_len)
    if kind == "transformer":
        return TransformerNet(n_endpoints, max_len)
    raise ValueError(f"unknown model kind: {kind}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
