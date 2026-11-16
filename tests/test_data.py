import pandas as pd
import pytest

from prefetchcache.data.coffee import load_coffee, parse_token, split_visits
from prefetchcache.data.retailrocket import sessions_from_events
from prefetchcache.data.sessions import RawSession, Vocab
from prefetchcache.data.split import split_sessions
from prefetchcache.data.synthetic import make_synthetic


def _tok(user, page, weekday, hour):
    return f"{user:07d}{page:03d}{weekday:02d}{hour:02d}"


def test_coffee_token_and_visit_split(tmp_path):
    assert parse_token("00000530010123") == ("0000053", 1, 23)
    assert split_visits([23, 23, 24, 30, 31, 5]) == [(0, 3), (3, 5), (5, 6)]
    assert split_visits([167, 0]) == [(0, 2)]  # Sunday 23:00 to Monday 00:00 is continuous
    line1 = [_tok(53, 1, 1, 23), _tok(53, 2, 1, 23), _tok(53, 2, 2, 0), _tok(53, 3, 2, 0),  # one visit, page 2 repeated
             _tok(53, 4, 3, 9), _tok(53, 5, 3, 9)]  # second visit, a day later
    line2 = [_tok(7, 1, 5, 10)]  # too short
    f = tmp_path / "coffee"
    f.write_text("".join(" -1 ".join(l) + " -1 -2\n" for l in (line1, line2)))
    sessions = load_coffee(f)
    assert [s.endpoints for s in sessions] == [
        ("GET /pages/1", "GET /pages/2", "GET /pages/3"),
        ("GET /pages/4", "GET /pages/5"),
    ]
    assert {s.user_id for s in sessions} == {"0000053"} and all(s.times is None for s in sessions)
    raw = load_coffee(f, split_on_time_gap=False, collapse_repeated_pages=False)
    assert len(raw) == 1 and len(raw[0].endpoints) == 6
    assert load_coffee(f, max_length=2)[0].endpoints == ("GET /pages/4", "GET /pages/5")


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
    # A changed user population no longer matches the frozen manifest.
    with pytest.raises(ValueError):
        split_sessions(sessions[5:], manifest=manifest)
