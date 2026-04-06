import pytest
from helpers import ROOT

from sysml.config import Config, load_config


def test_default_yaml_mirrors_dataclass_defaults():
    # configs/default.yaml documents the defaults; it must never drift from the code.
    assert load_config(ROOT / "configs" / "default.yaml") == Config()


def test_every_shipped_config_loads():
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        load_config(path)


def test_overrides_are_typed_and_nested():
    cfg = load_config(None, ["primary.type=logreg", "backtest.cost_bps=25", "regimes.causal=false",
                             "backtest.stress_periods=[]"])
    assert cfg.primary.type == "logreg"
    assert cfg.backtest.cost_bps == 25 and isinstance(cfg.backtest.cost_bps, int)
    assert cfg.regimes.causal is False
    assert cfg.backtest.stress_periods == []


def test_unknown_key_is_rejected():
    with pytest.raises(KeyError):
        load_config(None, ["primary.typo=1"])


def test_malformed_override_is_rejected():
    with pytest.raises(ValueError):
        load_config(None, ["no_equals_sign"])
