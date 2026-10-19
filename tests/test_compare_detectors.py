"""compare_detectors.py 纯度量函数的单元测试（不触网、不加载模型）。"""
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment"))

from compare_detectors import box_iou, mask_iou, label_matches_prompt, flatten_rows


class TestBoxIou:
    def test_identical(self):
        a = np.array([0, 0, 10, 10])
        assert box_iou(a, a) == pytest.approx(1.0)

    def test_partial_overlap(self):
        a = np.array([0, 0, 10, 10])
        b = np.array([5, 5, 15, 15])
        # 交集 25，并集 175
        assert box_iou(a, b) == pytest.approx(25 / 175)

    def test_disjoint(self):
        assert box_iou([0, 0, 5, 5], [10, 10, 15, 15]) == 0.0

    def test_one_inside_other(self):
        assert box_iou([0, 0, 10, 10], [2, 2, 8, 8]) == pytest.approx(36 / 100)

    def test_degenerate_zero_area(self):
        assert box_iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0.0


class TestMaskIou:
    def test_identical(self):
        m = (np.random.rand(32, 32) > 0.5).astype(np.uint8)
        assert mask_iou(m, m) == pytest.approx(1.0)

    def test_partial(self):
        m1 = np.zeros((4, 4), dtype=np.uint8); m1[:2] = 1   # 8 px
        m2 = np.zeros((4, 4), dtype=np.uint8); m2[1:3] = 1  # 8 px, 交 4
        assert mask_iou(m1, m2) == pytest.approx(4 / 12)

    def test_both_empty_convention(self):
        assert mask_iou(np.zeros((4, 4)), np.zeros((4, 4))) == 1.0

    def test_one_empty(self):
        assert mask_iou(np.zeros((4, 4)), np.ones((4, 4))) == 0.0

    def test_boolean_input(self):
        assert mask_iou(np.ones((4, 4), dtype=bool), np.ones((4, 4))) == pytest.approx(1.0)


class TestLabelMatches:
    def test_exact(self):
        assert label_matches_prompt("bear", "bear")

    def test_plural(self):
        assert label_matches_prompt("cow", "cows")

    def test_substring_blackswan(self):
        assert label_matches_prompt("swan", "blackswan")
        assert label_matches_prompt("bird", "blackswan")  # 经 PROMPT_SYNONYMS 语义匹配

    def test_non_match(self):
        assert not label_matches_prompt("car", "cows")
        assert not label_matches_prompt("dog", "bear")

    def test_multiword_label(self):
        assert label_matches_prompt("sports ball", "ball")


class TestFlattenRows:
    def test_flat_rows(self):
        rows = [{"target": "bear", "cross_iou": 0.87},
                {"target": "bear", "detector": "yolo11m", "mask_temporal_iou": 0.95}]
        flat = flatten_rows(rows)
        assert len(flat) == 2
        assert flat[0] == {"target": "bear", "detector": "-", "metric": "cross_iou", "value": 0.87}
        assert flat[1]["detector"] == "yolo11m"
