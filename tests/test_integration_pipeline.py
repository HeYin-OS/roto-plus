"""Integration tests: full pipeline on the real bear dataset.

Reuses the same building blocks as experiment/test_stroke.py but without its
interactive viewer / result-dumping side effects:

1. load real frames + real optical-flow cache + real mask cache + real stroke,
2. run candidate extraction + mask filtering (the "fitted" data prep),
3. propagate the stroke across all frames with flow + local_snapping,
4. assert shapes, finiteness, and that the result tracks the object.

Also verifies the RAFT cache can be recomputed on CUDA and stays consistent
with the cached tensor (marked slow — real model inference).
"""
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.integration

from utils.edge_snapping import (
    EdgeSnappingConfig,
    compute_all_candidates,
    local_snapping,
    resample_stroke_uniform,
)
from utils.kd_tree import BatchKDTree
from utils.path_utils import build_project_paths


@pytest.fixture(scope="module")
def bear_env(bear_data_paths, project_root):
    """Load config + verify all real-data inputs exist."""
    EdgeSnappingConfig.isConfigInit = False
    EdgeSnappingConfig.load(str(project_root / "config" / "snapping_init.yaml"))

    missing = [
        name
        for name, p in bear_data_paths.items()
        if name != "strokes" and not p.exists()
    ]
    if not bear_data_paths["strokes"]:
        missing.append("strokes")
    if missing:
        pytest.skip(f"bear dataset incomplete, missing: {missing}")

    return bear_data_paths


@pytest.fixture(scope="module")
def bear_images(bear_env):
    """Load all bear frames as uint8 RGB NHWC."""
    import cv2 as cv
    frame_paths = sorted(
        p for p in bear_env["frame_dir"].iterdir()
        if p.suffix.lower() in (".jpg", ".png")
    )
    assert len(frame_paths) > 10
    frames = np.stack([cv.imread(str(p), cv.IMREAD_COLOR) for p in frame_paths])
    frames = np.stack([cv.cvtColor(f, cv.COLOR_BGR2RGB) for f in frames])
    return frames


@pytest.fixture(scope="module")
def bear_candidates_filtered(bear_images, bear_env):
    """Candidate points, filtered by the mask band (same recipe as test_stroke.py)."""
    import cv2 as cv

    masks = torch.load(str(bear_env["mask_cache"])).numpy()
    if masks.ndim == 2:
        masks = masks[None, ...]

    all_candidates = compute_all_candidates(bear_images)
    # keep an unfiltered copy too
    unfiltered = [pts.copy() for pts in all_candidates]

    erode_size = EdgeSnappingConfig.erode_size
    dilate_size = EdgeSnappingConfig.dilate_size

    def erode(mask, t):
        if t <= 0:
            return mask
        m = (mask * 255).astype(np.uint8) if mask.dtype != np.uint8 else mask
        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (t * 2 + 1, t * 2 + 1))
        return cv.erode(m, k, iterations=1) / 255.0

    def dilate(mask, t):
        if t <= 0:
            return mask
        m = (mask * 255).astype(np.uint8) if mask.dtype != np.uint8 else mask
        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (t * 2 + 1, t * 2 + 1))
        return cv.dilate(m, k, iterations=1) / 255.0

    def filter_by_mask(points, mask, keep_inside):
        if len(points) == 0:
            return points
        H, W = mask.shape
        xs = np.clip(points[:, 0].astype(np.int32), 0, W - 1)
        ys = np.clip(points[:, 1].astype(np.int32), 0, H - 1)
        values = mask[ys, xs]
        return points[(values > 0.5) if keep_inside else (values < 0.5)]

    filtered = []
    for i, pts in enumerate(all_candidates):
        if i >= masks.shape[0]:
            filtered.append(pts)
            continue
        band_out = filter_by_mask(pts, erode(masks[i], erode_size), keep_inside=False)
        filtered.append(filter_by_mask(band_out, dilate(masks[i], dilate_size), keep_inside=True))

    return filtered, unfiltered, masks


def test_frames_loaded(bear_images):
    n, h, w, c = bear_images.shape
    assert c == 3
    assert h > 0 and w > 0
    assert n >= 50  # bear sequence has 82 frames


def test_flow_cache_shape_matches_frames(bear_images, bear_env):
    flow = torch.load(str(bear_env["flow_cache"]))
    n, h, w, _ = bear_images.shape
    assert flow.shape == (n - 1, h, w, 2), f"unexpected flow cache shape {flow.shape}"
    assert torch.isfinite(flow).all()


def test_mask_cache_shape_matches_frames(bear_images, bear_env):
    masks = torch.load(str(bear_env["mask_cache"]))
    n, h, w, _ = bear_images.shape
    assert masks.shape == (n, h, w), f"unexpected mask cache shape {masks.shape}"


def test_candidate_filtering_keeps_band_points(bear_candidates_filtered):
    filtered, unfiltered, masks = bear_candidates_filtered
    # filtered groups are subsets of the unfiltered ones in count
    for f, u in zip(filtered, unfiltered):
        assert len(f) <= len(u)
        assert f.shape[-1] == 2


def test_fitted_propagation_full_chain(bear_images, bear_env, bear_candidates_filtered):
    """The main pipeline: stroke -> snap on frame 0 -> flow+snap propagate across all frames."""
    filtered, unfiltered, masks = bear_candidates_filtered
    n, h, w, _ = bear_images.shape

    flow = torch.load(str(bear_env["flow_cache"])).numpy()
    kd_filtered = BatchKDTree(filtered)
    kd_unfiltered = BatchKDTree(unfiltered)

    stroke = np.load(str(bear_env["strokes"][0])).astype(np.float32)
    assert stroke.ndim == 2 and stroke.shape[1] == 2 and len(stroke) >= 5

    original_length = float(np.linalg.norm(np.diff(stroke, axis=0), axis=1).sum())

    # --- frame 0: snap the reference stroke ---
    candidates0 = kd_filtered.query_batch(0, stroke, EdgeSnappingConfig.r_s)
    fitted = local_snapping(
        stroke, bear_images[0], candidates0,
        previous_snapped_stroke=None,
        original_stroke_length=original_length,
    ).astype(np.float32)
    assert fitted.shape[1] == 2
    assert np.isfinite(fitted).all()
    # topology fix may merge/split points, so compare via arclength-aligned
    # resampling rather than point-to-point
    resampled = resample_stroke_uniform(fitted, len(stroke))
    assert np.linalg.norm(resampled - stroke, axis=1).mean() < 30.0

    # --- propagate ---
    prev = fitted
    trajectories = [fitted]
    for i_frame in range(1, n):
        xs = np.clip(prev[:, 0].astype(np.int32), 0, w - 1)
        ys = np.clip(prev[:, 1].astype(np.int32), 0, h - 1)
        moved = prev + flow[i_frame - 1, ys, xs]

        candidates = kd_filtered.query_batch(i_frame, moved, EdgeSnappingConfig.r_s)
        prev = local_snapping(
            moved, bear_images[i_frame], candidates,
            previous_snapped_stroke=prev if i_frame > 1 else None,
            original_stroke_length=original_length,
        ).astype(np.float32)

        assert prev.shape[1] == 2
        assert np.isfinite(prev).all()
        # every point must stay inside the frame
        assert prev[:, 0].min() >= -1 and prev[:, 0].max() <= w
        assert prev[:, 1].min() >= -1 and prev[:, 1].max() <= h
        trajectories.append(prev)

    assert len(trajectories) == n

    # frame-to-frame motion should be small (stroke tracks the object, no teleporting)
    # point counts drift between frames (topology fix merges/splits points), so
    # measure physical displacement with a symmetric Chamfer distance, which is
    # invariant to point count and parameterization
    def chamfer(a, b):
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
        return 0.5 * (d.min(axis=1).mean() + d.min(axis=0).mean())

    for i in range(1, n):
        gap = chamfer(trajectories[i], trajectories[i - 1])
        assert gap < 25.0, f"frame {i}: implausible jump {gap:.1f}px"


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_raft_recompute_matches_cache(bear_images, bear_env):
    """Recompute optical flow on CUDA for the first frame pair and compare to cache."""
    from reasoning.raft_predictor import RAFTPredictor
    predictor = RAFTPredictor(model_size="small")

    frames = bear_images[:2].astype(np.float32) / 255.0
    f1 = torch.from_numpy(frames[0]).permute(2, 0, 1)
    f2 = torch.from_numpy(frames[1]).permute(2, 0, 1)
    flow_fresh = predictor.compute_optical_flow_single(f1, f2)

    cache = torch.load(str(bear_env["flow_cache"]))
    flow_cached = cache[0]

    # large-model cache vs small-model recompute: direction should agree broadly
    magnitude_fresh = torch.linalg.norm(flow_fresh, dim=-1).numpy().flatten()
    magnitude_cached = torch.linalg.norm(flow_cached, dim=-1).numpy().flatten()
    assert float(magnitude_fresh.mean()) > 0.0
    # correlation of magnitudes over the whole frame
    corr = float(np.corrcoef(magnitude_fresh, magnitude_cached)[0, 1])
    assert corr > 0.3, f"fresh vs cached flow magnitude correlation too low: {corr}"
