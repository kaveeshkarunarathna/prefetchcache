"""Train a predictor, calibrate it on validation data and export it to ONNX.

    python -m prefetchcache.models.train --dataset synthetic --model lstm --seed 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from ..data.sessions import Session, Vocab
from .calibration import expected_calibration_error, fit_temperature, nll, softmax
from .nets import build_model, count_parameters
from .windows import IGNORE, make_windows


def _logits_and_targets(model: nn.Module, x: np.ndarray, y: np.ndarray, batch: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            outs.append(model(torch.from_numpy(x[i : i + batch])).numpy())
    logits = np.concatenate(outs)
    mask = y != IGNORE
    return logits[mask], y[mask]


def train_model(
    kind: str,
    train: Sequence[Session],
    val: Sequence[Session],
    n_endpoints: int,
    max_len: int = 20,
    seed: int = 0,
    epochs: int = 30,
    patience: int = 3,
    batch_size: int = 256,
    lr: float = 1e-3,
    verbose: bool = False,
) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(kind, n_endpoints, max_len)
    xt, yt = make_windows(train, max_len, model.pad_id)
    xv, yv = make_windows(val, max_len, model.pad_id)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE)
    best, best_state, stale = float("inf"), None, 0
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(xt))
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            logits = model(torch.from_numpy(xt[idx]))
            loss = loss_fn(logits.reshape(-1, n_endpoints), torch.from_numpy(yt[idx]).reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val_logits, val_targets = _logits_and_targets(model, xv, yv)
        val_loss = nll(val_logits, val_targets)
        if verbose:
            print(f"  epoch {epoch + 1}: val nll {val_loss:.4f}")
        if val_loss < best - 1e-4:
            best, stale = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    val_logits, val_targets = _logits_and_targets(model, xv, yv)
    temperature = fit_temperature(val_logits, val_targets)
    info = {
        "model": kind,
        "seed": seed,
        "max_len": max_len,
        "pad_id": model.pad_id,
        "n_endpoints": n_endpoints,
        "parameters": count_parameters(model),
        "epochs_run": epoch + 1,
        "train_seconds": round(time.perf_counter() - t0, 1),
        "temperature": temperature,
        "val_top1_accuracy": float((val_logits.argmax(1) == val_targets).mean()),
        "val_nll": nll(val_logits, val_targets),
        "val_nll_calibrated": nll(val_logits, val_targets, temperature),
        "val_ece": expected_calibration_error(softmax(val_logits), val_targets),
        "val_ece_calibrated": expected_calibration_error(softmax(val_logits, temperature), val_targets),
    }
    return model, info


class _LastStep(nn.Module):
    """Serving graph: logits for the next request only."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)[:, -1, :]


def export_onnx(model: nn.Module, info: dict, vocab: Vocab, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    example = torch.full((1, info["max_len"]), info["pad_id"], dtype=torch.long)
    torch.onnx.export(
        _LastStep(model).eval(),
        (example,),
        str(out / "model.onnx"),
        input_names=["history"],
        output_names=["logits"],
        dynamic_axes={"history": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    (out / "model.json").write_text(json.dumps({**info, "vocab": vocab.names}, indent=1))
    return out


def main() -> None:
    from ..experiments.common import load_dataset  # local import keeps torch out of the sweep path

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synthetic", "coffee", "retailrocket"], default="synthetic")
    ap.add_argument("--path")
    ap.add_argument("--model", choices=["lstm", "transformer"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--results", default="results/baselines")
    args = ap.parse_args()

    data = load_dataset(args.dataset, args.path, args.min_count, args.results)
    model, info = train_model(args.model, data.train, data.val, len(data.vocab), args.max_len, args.seed, args.epochs, verbose=True)
    out = export_onnx(model, info, data.vocab, Path(args.out) / args.dataset / args.model / f"seed{args.seed}")
    print(json.dumps({k: v for k, v in info.items()}, indent=1))
    print(f"exported to {out}")


if __name__ == "__main__":
    main()
