import pytest
import torch

import thermano.models  # noqa: F401  (registers components)
import thermano.physics  # noqa: F401
from thermano.config import DomainConfig, LaserConfig, RunConfig


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    yield
    torch.set_default_dtype(old)


@pytest.fixture
def domain():
    return DomainConfig()


@pytest.fixture
def laser():
    return LaserConfig(n_tracks=3, hatch_spacing=0.2, start=(0.5, 0.3))


@pytest.fixture
def tiny_cfg(tmp_path):
    """A physics run small enough for CI (seconds on a laptop CPU)."""
    from thermano.config import EvaluationConfig, ModelConfig, SamplingConfig, TrainingConfig

    return RunConfig(
        name="tiny", mode="physics", seed=1, dtype="float64", output_dir=str(tmp_path),
        laser=LaserConfig(n_tracks=1, scenarios=((0,),), sensors_per_track=4),
        model=ModelConfig(type="en_deeponet", branch_layers=(16,), trunk_layers=(16, 16), latent_dim=8),
        sampling=SamplingConfig(num_time_slices=4, points_per_slice=40, initial_points=80,
                                boundary_points_per_face=20, cluster_fraction=0.0),
        training=TrainingConfig(optimizer="adam", learning_rate=1e-3, max_epochs=15, log_every=5),
        evaluation=EvaluationConfig(grid=(9, 5, 5), snapshot_fractions=(0.5,)),
    ).validate()
