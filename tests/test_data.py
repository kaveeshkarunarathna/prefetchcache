import pandas as pd
import pytest

from prefetchcache.data.coffee import load_coffee, parse_spmf_sequence
from prefetchcache.data.retailrocket import sessions_from_events
from prefetchcache.data.sessions import RawSession, Vocab
from prefetchcache.data.split import split_sessions
from prefetchcache.data.synthetic import make_synthetic


def test_spmf_parsing(tmp_path):
    assert parse_spmf_sequence("3 -1 7 -1 3 -1 -2") == [3, 7, 3]
    f = tmp_path / "seq.txt"
    f.write_text("1 -1 2 -1 -2\n5 -1 -2\n\n4 -1 4 -1 9 -1 -2\n")
    sessions = load_coffee(f)
    assert [s.endpoints for s in sessions] == [
        ("GET /pages/1", "GET /pages/2"),
        ("GET /pages/4", "GET /pages/4", "GET /pages/9"),
    ]  # single-page sequence dropped
    assert all(s.times is None for s in sessions)


def test_retailrocket_sessionisation():
    df = pd.DataFrame(
        {
            "timestamp": [0, 60_000, 4_000_000, 4_010_000, 5_000, 6_000],
            "visitorid": [1, 1, 1, 1, 2, 2],
            "event": ["view", "addtocart", "view", "transaction", "view", "view"],
            "itemid": [10, 10, 11, 11, 10, 12],
        }
    )
    sessions = sessions_from_events(df)
    assert len(sessions) == 3
    assert sessions[0].endpoints == ("GET /items/10", "POST /cart")
    assert sessions[1].endpoints == ("GET /items/11", "POST /orders")
    assert sessions[2].times == (5.0, 6.0)


def test_vocab_unknown_and_cacheable():
    train = [RawSession("u", "s", ("GET /a", "POST /b", "GET /a"))]
    vocab = Vocab.fit(train)
    encoded = vocab.encode([RawSession("v", "t", ("GET /a", "GET /new"))])[0]
    assert encoded.endpoints == (vocab.index["GET /a"], 0)
    assert vocab.cacheable[vocab.index["GET /a"]]
    assert not vocab.cacheable[vocab.index["POST /b"]]
    assert not vocab.cacheable[0]


def test_split_is_user_level_and_frozen(tmp_path):
    sessions = make_synthetic(n_users=200, seed=1)
    manifest = tmp_path / "split.json"
    a = split_sessions(sessions, seed=0, manifest=manifest)
    users = [{s.user_id for s in part} for part in (a.train, a.val, a.test)]
    assert not (users[0] & users[1]) and not (users[0] & users[2]) and not (users[1] & users[2])
    assert len(users[0]) == 140 and len(users[1]) == 30 and len(users[2]) == 30
    # A different seed must not change a frozen split.
    b = split_sessions(sessions, seed=99, manifest=manifest)
    assert [s.session_id for s in b.test] == [s.session_id for s in a.test]
    # Tampering is detected.
    manifest.write_text(manifest.read_text().replace('"u0"', '"uX"'))
    with pytest.raises(ValueError):
        split_sessions(sessions, manifest=manifest)
