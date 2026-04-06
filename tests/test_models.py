import numpy as np
import pytest

from sysml.config import MetaConfig, PrimaryConfig
from sysml.models import (
    LogisticPrimary,
    MetaModel,
    balanced_class_weights,
    expanding_window_oof,
    make_primary,
    make_sequences,
    select_threshold,
)


def test_sequences_end_at_the_decision_row():
    x = np.arange(40, dtype=float).reshape(20, 2)
    seq = make_sequences(x, 5)
    assert seq.shape == (16, 5, 2) and seq.dtype == np.float32
    for j in (0, 7, 15):
        assert (seq[j] == x[j : j + 5]).all()
        assert (seq[j][-1] == x[j + 4]).all()  # sample j is decided at row j + seq_len - 1


def test_sequences_need_enough_rows():
    with pytest.raises(ValueError):
        make_sequences(np.zeros((3, 2)), 5)


def test_balanced_class_weights():
    w = balanced_class_weights(np.array([1] * 75 + [0] * 25))
    assert w[0] > 1 > w[1] and w[0] * 25 == pytest.approx(w[1] * 75)
    assert balanced_class_weights(np.ones(10)) == {0: 1.0, 1: 1.0}


class _Recorder:
    """Stand-in model that records which sample ids it trained on / predicted."""

    train_max: list[int] = []
    val_min: list[int] = []
    history = None

    def fit(self, x, y, class_weight=None, epochs=None):
        _Recorder.train_max.append(int(x[:, 0, 0].max()))
        return self

    def predict_proba(self, x):
        _Recorder.val_min.append(int(x[:, 0, 0].min()))
        return np.full(len(x), 0.5)


def test_expanding_window_oof_is_purged_and_forward_only():
    n, gap = 600, 40
    x = np.zeros((n, 3, 1), dtype=np.float32)
    x[:, 0, 0] = np.arange(n)  # sample id
    y = (np.arange(n) % 2).astype(int)
    _Recorder.train_max, _Recorder.val_min = [], []
    oof, folds = expanding_window_oof(_Recorder, x, y, n_splits=5, gap=gap)

    assert len(folds) == 5
    assert np.isnan(oof[: n // 6]).all()  # the first block is never predicted
    assert np.isfinite(oof[n // 6 :]).all()
    for tr_max, va_min in zip(_Recorder.train_max, _Recorder.val_min):
        assert va_min - tr_max > gap  # label look-ahead cannot reach the validation block
    assert folds["n_train"].is_monotonic_increasing  # expanding, not rolling


def test_logistic_primary_learns_a_separable_signal():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((800, 6, 4)).astype(np.float32)
    y = (x[:, -1, 0] > 0).astype(int)
    model = make_primary(PrimaryConfig(type="logreg"), n_features=4, seed=0).fit(x[:600], y[:600])
    assert ((model.predict_proba(x[600:]) > 0.5) == y[600:]).mean() > 0.9
    assert isinstance(model, LogisticPrimary)


def test_make_primary_rejects_unknown_type():
    with pytest.raises(ValueError):
        make_primary(PrimaryConfig(type="transformer"), 4, 0)


def test_threshold_ignores_tiny_support():
    cfg = MetaConfig()
    rng = np.random.default_rng(0)
    proba = rng.uniform(0.3, 0.6, 400)
    y = (rng.uniform(size=400) < 0.4).astype(int)
    y[np.argsort(proba)[-3:]] = 1  # the three most confident calls happen to be right
    thr, prec, n = select_threshold(proba, y, cfg)
    assert n >= 40  # at least 10% of the hold-out survives: a 100%-precision fluke is not chosen
    assert prec < 0.99


def test_threshold_falls_back_to_default_without_feasible_support():
    thr, prec, n = select_threshold(np.full(50, 0.1), np.zeros(50, dtype=int), MetaConfig())
    assert thr == MetaConfig().default_threshold and n == 0


def test_meta_model_learns_and_is_scored_out_of_sample():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((1500, 5))
    y = (x[:, 0] + 0.8 * rng.standard_normal(1500) > 0.4).astype(int)
    cfg = MetaConfig(n_estimators=60)
    meta = MetaModel(cfg, seed=0).fit(x, y, embargo=3)
    fresh = rng.standard_normal((800, 5))
    y_fresh = (fresh[:, 0] + 0.8 * rng.standard_normal(800) > 0.4).astype(int)
    p = meta.predict_proba(fresh)
    assert p[y_fresh == 1].mean() > p[y_fresh == 0].mean() + 0.1
    assert 0.0 <= meta.val_precision_ <= 1.0 and meta.val_support_ > 0
    assert meta.val_support_ <= int(1500 * cfg.val_fraction)  # scored on the hold-out only


def test_meta_model_refuses_tiny_or_single_class_data():
    with pytest.raises(ValueError):
        MetaModel(MetaConfig()).fit(np.zeros((30, 3)), np.zeros(30, dtype=int))
    with pytest.raises(ValueError):
        MetaModel(MetaConfig()).fit(np.random.default_rng(0).normal(size=(500, 3)),
                                    np.zeros(500, dtype=int))


@pytest.mark.tf
def test_lstm_primary_end_to_end_on_random_data():
    pytest.importorskip("tensorflow")
    from sysml.config import LSTMConfig

    cfg = PrimaryConfig(type="lstm", seq_len=8, lstm=LSTMConfig(units1=8, units2=4, dense=4,
                                                                epochs_final=2, batch_size=32))
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 8, 5)).astype(np.float32)
    y = (x[:, -1, 0] > 0).astype(int)
    model = make_primary(cfg, n_features=5, seed=0).fit(x, y, class_weight={0: 1.0, 1: 1.0})
    p = model.predict_proba(x[:10])
    assert p.shape == (10,) and ((0 <= p) & (p <= 1)).all()
    assert model.history is not None and "loss" in model.history
