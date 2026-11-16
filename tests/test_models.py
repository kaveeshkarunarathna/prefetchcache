import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")

from prefetchcache.data.sessions import Session, Vocab
from prefetchcache.data.split import split_sessions
from prefetchcache.data.synthetic import make_synthetic
from prefetchcache.models.calibration import expected_calibration_error, fit_temperature, nll, softmax
from prefetchcache.models.nets import build_model
from prefetchcache.models.train import _LastStep, export_onnx, train_model
from prefetchcache.models.windows import IGNORE, encode_history, make_windows
from prefetchcache.predictors.onnx_predictor import OnnxPredictor


def test_windows_are_left_padded_next_step_pairs():
    x, y = make_windows([Session("u", "s", (1, 2, 3))], max_len=4, pad_id=9)
    assert x.tolist() == [[9, 9, 1, 2]] and y.tolist() == [[IGNORE, IGNORE, 2, 3]]
    x, y = make_windows([Session("u", "s", tuple(range(1, 8)))], max_len=4, pad_id=9)
    assert sorted(map(tuple, x.tolist())) == [(3, 4, 5, 6), (9, 9, 1, 2)]
    assert encode_history([1, 2, 3, 4, 5], 3, 9).tolist() == [3, 4, 5]


@pytest.mark.parametrize("kind", ["lstm", "transformer"])
def test_models_are_causal(kind):
    torch.manual_seed(0)
    model = build_model(kind, 10, max_len=6).eval()
    a = torch.tensor([[1, 2, 3, 4, 5, 6]])
    b = torch.tensor([[1, 2, 3, 4, 9, 9]])
    with torch.no_grad():
        assert torch.allclose(model(a)[0, :4], model(b)[0, :4], atol=1e-5)


def test_temperature_scaling_reduces_overconfidence():
    rng = np.random.default_rng(0)
    targets = rng.integers(0, 5, 4000)
    logits = rng.normal(size=(4000, 5))
    logits[np.arange(4000), targets] += 1.0
    hot = logits * 5  # overconfident
    t = fit_temperature(hot, targets)
    assert 3.5 < t < 6.5
    assert nll(hot, targets, t) < nll(hot, targets)
    assert expected_calibration_error(softmax(hot, t), targets) < expected_calibration_error(softmax(hot), targets)


@pytest.mark.parametrize("kind", ["lstm", "transformer"])
def test_train_export_and_serve(kind, tmp_path):
    split = split_sessions(make_synthetic(n_users=300, n_pages=12, seed=2), seed=0)
    vocab = Vocab.fit(split.train)
    train, val = vocab.encode(split.train), vocab.encode(split.val)
    model, info = train_model(kind, train, val, len(vocab), max_len=8, epochs=6)
    assert info["val_top1_accuracy"] > 0.3  # chance is about 0.08
    predictor = OnnxPredictor(export_onnx(model, info, vocab, tmp_path / kind))
    history = list(val[0].endpoints[:5])
    ranked = predictor.predict(history, top_n=3)
    probs = [p for _, p in ranked]
    assert len(ranked) == 3 and probs == sorted(probs, reverse=True) and 0 < sum(probs) <= 1 + 1e-6
    # ONNX output matches PyTorch.
    x = torch.from_numpy(encode_history(history, 8, info["pad_id"])[None])
    with torch.no_grad():
        ref = torch.softmax(_LastStep(model)(x)[0] / info["temperature"], dim=0).numpy()
    assert ranked[0][0] == int(ref.argmax()) and abs(ranked[0][1] - float(ref.max())) < 1e-4
