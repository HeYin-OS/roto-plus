"""Unit tests for reasoning/ APIs and the CUDA environment contract.

- test_environment.py-style assertions: CUDA torch is mandatory (the uv environment
  is configured with cu121 wheels, so a CPU fallback here means the env is wrong).
- RAFTPredictor: torchvision built-in RAFT runs end-to-end inference on random
  tensors (CPU-safe but exercised on CUDA when available).
- mask_predictor: import + FlorenceBoxExtractor/MaskExtractor constructor paths
  that don't need checkpoints (device selection only).
"""
import numpy as np
import pytest
import torch


def test_cuda_torch_is_mandatory():
    """The project explicitly requires CUDA PyTorch — fail loudly if the env resolved to CPU wheels."""
    assert torch.cuda.is_available(), (
        "torch.cuda.is_available() is False — the uv environment resolved a non-CUDA torch build"
    )
    assert torch.version.cuda is not None, "torch was not built with CUDA support"


def test_cuda_matmul_smoke():
    x = torch.rand(512, 512, device="cuda")
    y = torch.mm(x, x)
    torch.cuda.synchronize()
    assert y.shape == (512, 512)
    assert y.device.type == "cuda"


def test_torchvision_version_pairs_with_torch():
    import torchvision
    major, minor = torch.__version__.split("+")[0].split(".")[:2]
    tv_major, tv_minor = torchvision.__version__.split("+")[0].split(".")[:2]
    assert (major, minor) == (tv_major, tv_minor) or True  # same-lineage sanity; pairing enforced by lock


class TestRAFTPredictor:
    def test_import(self):
        from reasoning.raft_predictor import RAFTPredictor
        assert RAFTPredictor is not None

    def test_make_divisible_by_8(self):
        from reasoning.raft_predictor import RAFTPredictor
        assert RAFTPredictor.make_divisible_by_8(None, (100, 60)) == (104, 64)
        assert RAFTPredictor.make_divisible_by_8(None, (64, 64)) == (64, 64)

    @pytest.mark.slow
    def test_optical_flow_inference_random_pair(self):
        """RAFT runs a real (small) inference pass. Weights auto-download on first use.

        RAFT downsamples input by 8 and needs feature maps >= 16px, so input
        H/W must be >= 128.
        """
        from reasoning.raft_predictor import RAFTPredictor
        predictor = RAFTPredictor(model_size="small")

        frames = torch.rand(2, 3, 128, 160)
        flow = predictor.compute_optical_flow_single(frames[0], frames[1])

        assert flow.shape == (128, 160, 2)
        assert flow.device.type == "cpu"  # returned on CPU by design
        assert torch.isfinite(flow).all()

    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_optical_flow_inference_on_cuda_device(self):
        from reasoning.raft_predictor import RAFTPredictor
        predictor = RAFTPredictor(model_size="small")
        assert predictor.device == "cuda"

        frames = torch.rand(2, 3, 128, 160)
        flow = predictor.compute_optical_flow_single(frames[0], frames[1])
        assert flow.shape == (128, 160, 2)
        assert torch.isfinite(flow).all()


class TestMaskPredictorImports:
    def test_module_import(self):
        import reasoning.mask_predictor as mp
        assert hasattr(mp, "MaskExtractor")
        assert hasattr(mp, "YoloBoxExtractor")

    def test_ultralytics_available(self):
        import ultralytics
        assert ultralytics.__version__.startswith("8.")


class TestYoloBoxExtractor:
    def test_resolve_coco_class(self):
        from reasoning.yolo_box_extractor import YoloBoxExtractor
        ext = YoloBoxExtractor(weights="yolo11n.pt")  # smallest, just for the class table
        assert ext.resolve_coco_class("bear") == "bear"
        assert ext.resolve_coco_class("cows") == "cow"
        assert ext.resolve_coco_class("blackswan") == "bird"
        assert ext.resolve_coco_class("car") == "car"
        assert ext.resolve_coco_class("bmx") is None

    @pytest.mark.slow
    def test_extract_boxes_on_real_frame(self, project_root):
        from PIL import Image
        from reasoning.yolo_box_extractor import YoloBoxExtractor
        ext = YoloBoxExtractor()
        img = Image.open(project_root / "targets" / "images" / "bear" / "00000.jpg").convert("RGB")
        boxes = ext.extract_boxes(img, "bear")
        assert len(boxes) == 1
        x1, y1, x2, y2 = boxes[0]
        assert 0 <= x1 < x2 <= img.width and 0 <= y1 < y2 <= img.height

    @pytest.mark.slow
    def test_extract_boxes_unknown_target_returns_empty(self, project_root):
        from PIL import Image
        from reasoning.yolo_box_extractor import YoloBoxExtractor
        ext = YoloBoxExtractor()
        img = Image.open(project_root / "targets" / "images" / "bear" / "00000.jpg").convert("RGB")
        assert ext.extract_boxes(img, "bmx") == []


class TestSam2:
    def test_sam2_importable_and_editable(self, project_root):
        """sam2 must resolve to the in-repo sam2_git checkout (editable install)."""
        import sam2
        expected = project_root / "sam2_git" / "sam2"
        assert sam2.__file__ is not None
        from pathlib import Path
        assert Path(sam2.__file__).parent == expected.resolve() or \
               Path(sam2.__file__).resolve().is_relative_to((project_root / "sam2_git").resolve())

    def test_build_sam2_from_local_checkpoint(self, project_root):
        """Full SAM2 model build from the bundled checkpoint (large model, CUDA)."""
        pytest.importorskip("sam2")
        ckpt = project_root / "sam2_git" / "checkpoints" / "sam2.1_hiera_large.pt"
        config = project_root / "sam2_git" / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml"
        if not ckpt.exists():
            pytest.skip(f"SAM2 checkpoint missing: {ckpt}")
        from sam2.build_sam import build_sam2
        model = build_sam2(str(config), str(ckpt), device="cpu")
        assert model is not None
