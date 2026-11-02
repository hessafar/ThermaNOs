"""Scan paths, sampling and dataset I/O."""

import math

import numpy as np
import pytest

from thermano.config import SamplingConfig
from thermano.data.sampling import boundary_points, initial_points, interior_points
from thermano.data.scan_paths import ScanPath, all_scenarios, sequential_pinn_count, unique_prefixes
from thermano.data.snapshots import OperatorDataset, read_dataset, write_dataset

RNG = np.random.default_rng(0)


def test_laser_moves_at_scan_speed(laser):
    p = ScanPath(laser, (0,))
    a, b = p.center(0.0), p.center(0.004)
    assert np.linalg.norm(b - a) == pytest.approx(laser.speed * 0.004)


def test_laser_jumps_to_next_track(laser):
    p = ScanPath(laser, (2, 0, 1))
    t = p.track_time
    np.testing.assert_allclose(p.center(0.0), [0.5, 0.3 + 2 * 0.2])
    np.testing.assert_allclose(p.center(t + 1e-12), [0.5, 0.3], atol=1e-8)
    np.testing.assert_allclose(p.center(2.5 * t), [0.5 + 0.5, 0.3 + 0.2])


def test_sensor_vector_length(laser):
    p = ScanPath(laser, (0, 1, 2))
    assert p.sensor_values().shape == (2 * (laser.sensors_per_track + 1) * 3,)


def test_invalid_order_rejected(laser):
    with pytest.raises(ValueError):
        ScanPath(laser, (0, 0, 1))


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_scenario_tree_counts(n):
    assert len(all_scenarios(n)) == math.factorial(n)
    assert len(unique_prefixes(n)) == sequential_pinn_count(n)


def test_paper_three_tracks_needs_fifteen_pinns():
    assert sequential_pinn_count(3) == 15  # paper Sec. 3.3


def test_sampling_reproducible_and_in_bounds(domain):
    s = SamplingConfig(num_time_slices=5, points_per_slice=64, cluster_fraction=0.5, cluster_center=1.0)
    a = interior_points(domain, 0.01, s, seed=3, cluster_center=1.0)
    b = interior_points(domain, 0.01, s, seed=3, cluster_center=1.0)
    np.testing.assert_array_equal(a, b)
    lo, hi = domain.bounds()
    assert (a[:, 1:] >= lo).all() and (a[:, 1:] <= hi).all()
    assert (a[:, 0] >= 0).all() and (a[:, 0] < 0.01).all()
    assert len(a) == 5 * 64 + 5 * 32


def test_clustering_refines_band(domain):
    """Extra points all fall in [c - w, c + w]; overall density there exceeds uniform."""
    s = SamplingConfig(num_time_slices=1, points_per_slice=2048, cluster_fraction=1.0,
                       cluster_width=0.3, cluster_gamma=2.0)
    pts = interior_points(domain, 0.01, s, seed=0, cluster_center=1.0)
    extra = pts[2048:, 1]
    assert np.all(np.abs(extra - 1.0) <= 0.3 + 1e-12)
    in_band = np.mean(np.abs(pts[:, 1] - 1.0) <= 0.3)
    assert in_band > 0.6  # uniform sampling alone would give 0.3


def test_initial_points_at_t0(domain):
    pts = initial_points(domain, SamplingConfig(initial_points=128), seed=0)
    assert np.all(pts[:, 0] == 0.0)


@pytest.mark.parametrize("face,axis,value,sign", [("z_max", 3, 1.0, 1), ("x_min", 1, 0.0, -1)])
def test_boundary_points_and_normals(domain, face, axis, value, sign):
    pts, nrm = boundary_points(domain, 0.01, face, 64, seed=0)
    assert np.all(pts[:, axis] == value)
    assert np.all(nrm[:, axis] == sign) and np.all(np.abs(nrm).sum(1) == 1)


def test_dataset_roundtrip_and_schema_check(tmp_path):
    ds = OperatorDataset(RNG.random((4, 3)), RNG.random((10, 2)), RNG.random((4, 10)))
    write_dataset(tmp_path / "d.h5", ds, units="test")
    back = read_dataset(tmp_path / "d.h5")
    np.testing.assert_array_equal(back.targets, ds.targets)
    b, c, y = back.subset([1, 3]).flatten()
    assert b.shape == (20, 3) and c.shape == (20, 2) and y.shape == (20, 1)

    bad = OperatorDataset(RNG.random((4, 3)), RNG.random((9, 2)), RNG.random((4, 10)))
    write_dataset(tmp_path / "bad.h5", bad)
    with pytest.raises(ValueError, match="inconsistent"):
        read_dataset(tmp_path / "bad.h5")
