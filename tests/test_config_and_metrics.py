from pathlib import Path

import numpy as np
import pytest

from thermano.config import load_config
from thermano.evaluation.metrics import mape, melt_pool_dimensions, relative_error

CONFIGS = sorted(Path(__file__).parents[1].joinpath("configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_shipped_configs_are_valid(path):
    cfg = load_config(path)
    assert cfg.name == path.stem


def test_config_hash_is_stable_and_sensitive():
    a = load_config(CONFIGS[0])
    assert a.hash() == load_config(CONFIGS[0]).hash()
    b = load_config(CONFIGS[0], {"training.max_epochs": 7})
    assert b.training.max_epochs == 7 and b.hash() != a.hash()


def test_unknown_key_is_an_error(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: x\nmode: physics\ntraining: {max_epoch: 3}\n")  # typo
    with pytest.raises(ValueError, match="max_epoch"):
        load_config(p)


def test_path_outside_domain_is_an_error(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: x\nmode: physics\nlaser: {track_length: 5.0}\n")
    with pytest.raises(ValueError, match="leaves the domain"):
        load_config(p)


def test_mape_and_relative_error():
    assert mape([110, 90], [100, 100]) == pytest.approx(10.0)
    assert relative_error(95, 100) == pytest.approx(5.0)


def test_melt_pool_dimensions_of_ellipsoid():
    x, y, z = np.linspace(0, 2, 401), np.linspace(0, 1, 201), np.linspace(0, 1, 201)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    a, b, c = 0.3, 0.15, 0.1  # semi-axes; pool centred on the top surface
    r = ((X - 1) / a) ** 2 + ((Y - 0.5) / b) ** 2 + ((Z - 1) / c) ** 2
    T = 2000.0 - 500.0 * r
    dims = melt_pool_dimensions(T, x, y, z, T_melt=1500.0)
    assert dims["length"] == pytest.approx(2 * a, abs=2e-3)
    assert dims["width"] == pytest.approx(2 * b, abs=2e-3)
    assert dims["depth"] == pytest.approx(c, abs=2e-3)
