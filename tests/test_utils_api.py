"""Unit tests for the main project APIs in utils/ (pure logic, no GPU, no heavy models).

Covers:
- YamlUtil (utils/yaml_reader.py)
- build_project_paths / get_target_name (utils/path_utils.py)
- BatchKDTree (utils/kd_tree.py)
- EdgeSnappingConfig + edge-snapping helpers (utils/edge_snapping.py)
- edge_snapping_utils kernels / packing helpers
"""
import numpy as np
import pytest
import torch

from utils.yaml_reader import YamlUtil
from utils.path_utils import build_project_paths, get_target_name
from utils.kd_tree import BatchKDTree
from utils.edge_snapping import (
    EdgeSnappingConfig,
    compute_all_candidates,
    local_snapping,
    resample_stroke_uniform,
    detect_crossings,
    segments_intersect,
    detect_direction_issues,
    check_and_fix_topology,
    candidate_point_sets_defense,
    enforce_minimum_distance,
)
from utils.edge_snapping_utils import (
    create_gaussian_kernel,
    create_fdog_kernel,
    pack_jagged_list_to_array,
    slice_candidate_group_by_index,
    rgb_np_to_gray_tensor,
)


# ==================================================
# YamlUtil
# ==================================================

class TestYamlUtil:
    def test_read_project_config(self, project_root):
        config = YamlUtil.read(str(project_root / "config" / "test_video_init.yaml"))
        assert config["video"]["url_head"] == "./targets/images/bear"
        assert config["video"]["format"] == "jpg"

    def test_read_snapping_config_keys(self, project_root):
        config = YamlUtil.read(str(project_root / "config" / "snapping_init.yaml"))
        s = config["snapping"]
        for key in ("theta", "alpha", "sigma_c", "sigma_s", "sigma_m", "rho",
                    "r_s", "erode_size", "dilate_size"):
            assert key in s, f"missing snapping config key: {key}"

    def test_read_from_string(self):
        parsed = YamlUtil.read_from_string("a:\n  b: [1, 2]\n")
        assert parsed == {"a": {"b": [1, 2]}}

    def test_cache_and_clear(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("k: 1\n", encoding="utf-8")
        YamlUtil.clear_cache()
        assert YamlUtil.read(str(cfg)) == {"k": 1}
        # cached read returns same object identity
        assert YamlUtil.read(str(cfg)) is YamlUtil.read(str(cfg))
        YamlUtil.clear_cache(str(cfg))
        assert str(cfg) not in YamlUtil._cache


# ==================================================
# path utils
# ==================================================

class TestPathUtils:
    def test_build_project_paths(self, project_root):
        base, config_dir, cache_dir, frame_dir, result_dir, stroke_dir = build_project_paths()
        assert base == project_root
        assert config_dir == project_root / "config"
        assert cache_dir == project_root / "caches"
        assert frame_dir == project_root / "targets" / "images" / "bear"
        assert result_dir == project_root / "results"
        assert stroke_dir == project_root / "stroke"

    def test_frame_dir_exists_with_frames(self, project_root):
        _, _, _, frame_dir, _, _ = build_project_paths()
        frames = list(frame_dir.glob("*.jpg"))
        assert len(frames) > 0, "configured frame dir has no jpg frames"

    def test_get_target_name(self, tmp_path):
        assert get_target_name(tmp_path) == tmp_path.name


# ==================================================
# BatchKDTree
# ==================================================

class TestBatchKDTree:
    def _groups(self):
        rng = np.random.default_rng(0)
        return [rng.random((50, 2)).astype(np.float32) * 100 for _ in range(3)]

    def test_query_returns_points_within_radius(self):
        tree = BatchKDTree(self._groups())
        center = np.array([50.0, 50.0])
        result = tree.query(0, center, radius=10.0)
        assert result.ndim == 2 and result.shape[1] == 2
        assert np.all(np.linalg.norm(result - center, axis=1) <= 10.0 + 1e-6)

    def test_query_batch_shapes_and_bounds(self):
        groups = self._groups()
        tree = BatchKDTree(groups)
        centers = np.array([[10.0, 10.0], [50.0, 50.0], [95.0, 5.0]], dtype=np.float32)
        batches = tree.query_batch(1, centers, radius=20.0, max_candidates_per_point=5)
        assert len(batches) == 3
        for i, batch in enumerate(batches):
            assert batch.ndim == 2 and batch.shape[1] == 2
            assert len(batch) <= 5
            assert np.all(np.linalg.norm(batch - centers[i], axis=1) <= 20.0 + 1e-6)

    def test_query_batch_empty_neighborhood_falls_back_to_center(self):
        # points clustered far away from the query center -> no candidates in radius
        group = np.array([[1.0, 1.0], [1.5, 1.0]], dtype=np.float32)
        tree = BatchKDTree([group])
        batch = tree.query_batch(0, np.array([[90.0, 90.0]], dtype=np.float32), radius=2.0)
        assert batch[0].shape == (1, 2)
        np.testing.assert_allclose(batch[0][0], [90.0, 90.0])


# ==================================================
# EdgeSnappingConfig
# ==================================================

class TestEdgeSnappingConfig:
    def test_load_populates_all_fields(self, project_root):
        # reset singleton state so the test is independent of import order
        EdgeSnappingConfig.isConfigInit = False
        EdgeSnappingConfig.load(str(project_root / "config" / "snapping_init.yaml"))

        assert EdgeSnappingConfig.isConfigInit is True
        assert EdgeSnappingConfig.theta == 0.1
        assert EdgeSnappingConfig.alpha == 0.1
        assert EdgeSnappingConfig.r_s == 20
        assert EdgeSnappingConfig.max_stroke_points == 200
        assert EdgeSnappingConfig.enable_topology_fix is True

        # kernels are built at load time
        expected_x = 2 * EdgeSnappingConfig.X_MAX + 1
        expected_y = 2 * EdgeSnappingConfig.Y_MAX + 1
        assert EdgeSnappingConfig.fdog_kernel is not None
        assert EdgeSnappingConfig.gaussian_kernel is not None
        # fdog kernel is horizontal (1, X), gaussian kernel is vertical (Y, 1)
        assert EdgeSnappingConfig.fdog_kernel.shape == (1, expected_x)
        assert EdgeSnappingConfig.gaussian_kernel.shape == (expected_y, 1)

    def test_load_is_idempotent(self, project_root):
        EdgeSnappingConfig.isConfigInit = False
        EdgeSnappingConfig.load(str(project_root / "config" / "snapping_init.yaml"))
        theta_before = EdgeSnappingConfig.theta
        kernel_before = EdgeSnappingConfig.fdog_kernel
        EdgeSnappingConfig.load(str(project_root / "config" / "snapping_init.yaml"))
        assert EdgeSnappingConfig.theta == theta_before
        assert EdgeSnappingConfig.fdog_kernel is kernel_before


# ==================================================
# edge_snapping_utils helpers
# ==================================================

class TestEdgeSnappingUtils:
    def test_gaussian_kernel_normalized(self):
        kernel = create_gaussian_kernel(15, 3.0, 0)
        assert kernel.shape == (15, 1)
        np.testing.assert_allclose(kernel.sum(), 1.0, atol=1e-5)

    def test_fdog_kernel_is_dog(self):
        size = 9
        fdog = create_fdog_kernel(size, 1.0, 1.6, 0.99, 1)
        g1 = create_gaussian_kernel(size, 1.0, 1)
        g2 = create_gaussian_kernel(size, 1.6, 1)
        np.testing.assert_allclose(fdog, g1 - 0.99 * g2, atol=1e-6)

    def test_pack_jagged_and_slice_roundtrip(self):
        jagged = [
            np.array([[0, 0], [1, 1]], dtype=np.float32),
            np.array([[2, 2]], dtype=np.float32),
            np.array([], dtype=np.float32).reshape(0, 2),
            np.array([[3, 3], [4, 4], [5, 5]], dtype=np.float32),
        ]
        flat, ptr = pack_jagged_list_to_array(jagged)
        assert flat.shape == (6, 2)
        assert ptr.tolist() == [0, 2, 3, 3, 6]

        q0, q1 = slice_candidate_group_by_index(flat, ptr, 0)
        np.testing.assert_allclose(q0, jagged[0])
        np.testing.assert_allclose(q1, jagged[1])

        # slicing across the empty group yields empty arrays
        q2, q3 = slice_candidate_group_by_index(flat, ptr, 2)
        assert q2.shape[0] == 0
        np.testing.assert_allclose(q3, jagged[3])

    def test_rgb_np_to_gray_tensor(self):
        image = np.random.default_rng(1).integers(0, 255, (32, 48, 3), dtype=np.uint8)
        tensor = rgb_np_to_gray_tensor(torch.device("cpu"), image)
        assert tensor.shape == (1, 1, 32, 48)
        assert tensor.dtype == torch.float32
        assert float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0
        # white pixels -> ~1, black -> 0
        white = rgb_np_to_gray_tensor(torch.device("cpu"), np.full((4, 4, 3), 255, np.uint8))
        black = rgb_np_to_gray_tensor(torch.device("cpu"), np.zeros((4, 4, 3), np.uint8))
        assert float(white.max()) == 1.0
        assert float(black.max()) == 0.0


# ==================================================
# edge snapping geometry / topology helpers
# ==================================================

class TestGeometryHelpers:
    def test_segments_intersect(self):
        a = np.array([0.0, 0.0]); b = np.array([10.0, 10.0])
        c = np.array([0.0, 10.0]); d = np.array([10.0, 0.0])
        assert bool(segments_intersect(a, b, c, d)) is True
        # parallel non-touching
        e = np.array([0.0, 1.0]); f = np.array([10.0, 11.0])
        assert bool(segments_intersect(a, b, e, f)) is False

    def test_detect_crossings(self):
        # a "bowtie" polyline crosses itself
        bowtie = np.array([[0, 0], [10, 10], [10, 0], [0, 10]], dtype=np.float32)
        assert detect_crossings(bowtie) is True
        straight = np.array([[0, 0], [5, 0], [10, 0], [15, 0]], dtype=np.float32)
        assert detect_crossings(straight) is False

    def test_detect_direction_issues(self):
        original = np.array([[0, 0], [10, 0], [20, 0]], dtype=np.float32)
        same_dir = np.array([[0, 5], [10, 5], [20, 5]], dtype=np.float32)
        assert detect_direction_issues(same_dir, original) is False
        reversed_stroke = np.array([[20, 5], [10, 5], [0, 5]], dtype=np.float32)
        assert detect_direction_issues(reversed_stroke, original) is True

    def test_check_and_fix_topology_noop_on_clean_stroke(self):
        original = np.array([[0, 0], [10, 0], [20, 0], [30, 0]], dtype=np.float32)
        out = check_and_fix_topology(original.copy(), original)
        np.testing.assert_allclose(out, original)

    def test_resample_stroke_uniform(self):
        stroke = np.array([[0, 0], [10, 0], [30, 0]], dtype=np.float32)
        resampled = resample_stroke_uniform(stroke, 7)
        assert resampled.shape == (7, 2)
        # uniformly spaced along x, monotonic
        xs = resampled[:, 0]
        assert np.all(np.diff(xs) >= -1e-6)
        np.testing.assert_allclose(xs[0], 0.0, atol=1e-5)
        np.testing.assert_allclose(xs[-1], 30.0, atol=1e-5)

    def test_resample_degenerate_collapsed_stroke(self):
        # all points coincident -> total length ~0 -> tiled duplicate of the point
        collapsed = np.tile([5.0, 5.0], (3, 1)).astype(np.float32)
        out = resample_stroke_uniform(collapsed, 6)
        assert out.shape == (6, 2)
        np.testing.assert_allclose(out, np.tile([5.0, 5.0], (6, 1)))

    def test_resample_single_point_passthrough(self):
        # len < 2 strokes are returned unchanged by design
        point = np.array([[5.0, 5.0]])
        out = resample_stroke_uniform(point, 4)
        assert out is point

    def test_enforce_minimum_distance_caps_points(self):
        EdgeSnappingConfig.max_stroke_points = 200
        original = np.tile([0.0, 0.0], (5, 1)).astype(np.float32)
        long_stroke = np.stack([np.array([i, 0.0]) for i in range(50)], axis=0).astype(np.float32)
        out = enforce_minimum_distance(long_stroke, original)
        assert len(out) <= 200
        assert out.shape[1] == 2

    def test_candidate_point_sets_defense_pads_missing_groups(self):
        stroke = np.zeros((5, 2), dtype=np.float32)
        stroke[2] = [7.0, 8.0]
        candidates = [np.array([[1, 1]], dtype=np.float32), None]
        fixed = candidate_point_sets_defense(candidates, stroke)
        assert len(fixed) == 5
        # group 2 fell back to the stroke point
        np.testing.assert_allclose(fixed[2], [[7.0, 8.0]])
        # groups beyond the candidate list also fall back to stroke points
        np.testing.assert_allclose(fixed[4], [[0.0, 0.0]])


# ==================================================
# compute_all_candidates + local_snapping (CPU-safe smoke)
# ==================================================

class TestSnappingPipelineCPU:
    @pytest.fixture(autouse=True)
    def _load_config(self, project_root):
        EdgeSnappingConfig.isConfigInit = False
        EdgeSnappingConfig.load(str(project_root / "config" / "snapping_init.yaml"))

    def _synthetic_image(self, size=64):
        """A vertical black/white edge in the middle of the image."""
        img = np.zeros((size, size, 3), np.uint8)
        img[:, size // 2:] = 255
        return img

    def test_compute_all_candidates_finds_edge(self):
        img = self._synthetic_image()
        candidates = compute_all_candidates(np.stack([img]))
        assert len(candidates) == 1
        pts = candidates[0]
        assert pts.ndim == 2 and pts.shape[1] == 2
        assert pts.dtype == np.float32
        # all candidate x-coords sit near the vertical edge (within a few px)
        assert np.abs(pts[:, 0] - 32).max() <= 2.0
        assert len(pts) > 5

    def test_local_snapping_snaps_to_strong_edge(self):
        size = 64
        img = self._synthetic_image(size)
        # a wobbly stroke near the true edge at x = 32
        xs = 32 + np.sin(np.linspace(0, 3 * np.pi, 25)) * 4
        ys = np.linspace(4, size - 4, 25)
        stroke = np.stack([xs, ys], axis=1).astype(np.float32)

        candidate_pts = compute_all_candidates(np.stack([img]))[0]
        tree = BatchKDTree([candidate_pts])
        candidates_for_stroke = tree.query_batch(0, stroke, EdgeSnappingConfig.r_s)

        snapped = local_snapping(stroke, img, candidates_for_stroke)

        assert snapped.shape[1] == 2
        assert np.issubdtype(snapped.dtype, np.floating)
        # snapping should pull points closer to the edge than the input stroke
        before = np.abs(stroke[:, 0] - 32).mean()
        after = np.abs(snapped[:, 0] - 32).mean()
        assert after <= before, f"snapping did not improve edge adherence ({after:.2f} > {before:.2f})"

    def test_local_snapping_with_previous_stroke(self):
        # exercises the velocity/deform machinery when a previous frame is given
        size = 64
        img = self._synthetic_image(size)
        xs = 32 + np.sin(np.linspace(0, 2 * np.pi, 15)) * 3
        ys = np.linspace(4, size - 4, 15)
        stroke = np.stack([xs, ys], axis=1).astype(np.float32)
        previous = stroke + np.array([2.0, 0.0], dtype=np.float32)

        candidate_pts = compute_all_candidates(np.stack([img]))[0]
        tree = BatchKDTree([candidate_pts])
        candidates_for_stroke = tree.query_batch(0, previous, EdgeSnappingConfig.r_s)

        snapped = local_snapping(
            stroke, img, candidates_for_stroke,
            previous_snapped_stroke=previous,
            original_stroke_length=float(np.linalg.norm(np.diff(stroke, axis=0), axis=1).sum()),
        )
        assert snapped.shape[1] == 2
        assert len(snapped) >= 2
