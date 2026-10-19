"""Shared pytest fixtures: project root on sys.path, offscreen Qt, slow-marker."""
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: tests that run real model inference (RAFT/SAM2)")
    config.addinivalue_line("markers", "integration: full-pipeline tests on the real bear dataset")


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def bear_data_paths(project_root):
    """Real dataset paths used by integration tests (bear sequence)."""
    frame_dir = project_root / "targets" / "images" / "bear"
    return {
        "frame_dir": frame_dir,
        "flow_cache": project_root / "caches" / "flow" / "bear.pt",
        "mask_cache": project_root / "caches" / "mask" / "bear.pt",
        "strokes": sorted((project_root / "stroke" / "bear").glob("stroke_*.npy")),
    }


@pytest.fixture
def qt_offscreen(monkeypatch):
    """Force Qt offscreen platform before QApplication could be created."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    yield
