"""GT-free 双检测器对比实验（YOLO vs Florence-2）。

不依赖 DAVIS GT 标注（无需外部下载），用三层"可跑优先"的交叉度量：
  层1 box 对比    : 两检测器在首帧上的 box 交叉 IoU、置信度、推理时延
  层2 mask 对比   : 各 box 分别喂给同一个 SAM2 传播器（帧子集），
                   mask 序列交叉 IoU + 每检测器的时序稳定性（相邻帧 mask IoU）

设计约束（对应项目定位：算法工程，非科研）：
  - Florence-2 惰性加载，加载/下载失败自动降级为 YOLO 单检测器模式
  - 单目标失败只记录 warning，不中断整个实验
  - SAM2 mask 写入独立缓存 caches/mask/{target}_{detector}.pt，不碰主管线缓存
  - 结果输出到 results/detector_comparison/（metrics.csv + 首帧 box 叠加图）

用法：
  uv run python experiment/compare_detectors.py                    # 默认 bear
  uv run python experiment/compare_detectors.py --targets bear car # 多目标
  uv run python experiment/compare_detectors.py --max_frames 20 --skip_mask
"""
import sys
import time
import csv
import argparse
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

_file_path = Path(__file__).resolve()
_project_root = _file_path.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.path_utils import build_project_paths


# ----------------------------------------------------------------------------
# 纯度量函数（tests/test_compare_detectors.py 覆盖）
# ----------------------------------------------------------------------------

def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """两个 xyxy box 的 IoU。"""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    """两个二值 mask 的 IoU。双空 mask 约定为 1.0（同为"无目标"）。"""
    m1 = np.asarray(m1) > 0
    m2 = np.asarray(m2) > 0
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


# prompt 侧语义同义词：Florence 对这些场景常给出上位/下位标签
PROMPT_SYNONYMS = {
    "blackswan": {"swan", "bird"},
    "cows": {"cow", "cattle"},
    "ball": {"sports ball", "ball"},
}


def label_matches_prompt(label: str, prompt: str) -> bool:
    """检测标签与目标名是否匹配（大小写/单复数/子串/同义词容错）。

    例：("cow", "cows") / ("swan", "blackswan") / ("bird", "blackswan") 均匹配。
    """
    label = label.strip().lower()
    prompt = prompt.strip().lower()
    words = {w.strip() for w in label.replace("-", " ").split() if w.strip()}
    keys = {prompt, prompt.rstrip("s")} | PROMPT_SYNONYMS.get(prompt, set())
    for key in keys:
        if not key:
            continue
        if key in words:
            return True
        # blackswan <-> swan：任一方向子串命中即可
        for w in words:
            if key in w or w in key:
                return True
    return False


# ----------------------------------------------------------------------------
# 检测器适配层（统一接口: detect(image, prompt) -> (box_xyxy or None, conf, ms)）
# ----------------------------------------------------------------------------

class YoloAdapter:
    name = "yolo11m"

    def __init__(self, weights: str = "yolo11m.pt"):
        from reasoning.yolo_box_extractor import YoloBoxExtractor
        self.extractor = YoloBoxExtractor(weights=weights)
        # 预热一次（含 CUDA kernel 首载），后续计时才公平
        dummy = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
        self.extractor.extract_boxes(dummy, "car")

    def detect(self, image: Image.Image, prompt: str):
        """返回 (box_xyxy | None, conf | None, ms)。"""
        coco = self.extractor.resolve_coco_class(prompt)
        if coco is None:
            return None, None, 0.0
        class_ids = [cid for cid, name in self.extractor.model.names.items()
                     if name.lower() == coco]
        t0 = time.perf_counter()
        results = self.extractor.model.predict(
            source=np.asarray(image), conf=self.extractor.conf,
            classes=class_ids, device=self.extractor.device, verbose=False,
        )
        ms = (time.perf_counter() - t0) * 1000
        boxes = results[0].boxes if results else None
        if boxes is None or len(boxes) == 0:
            return None, None, ms
        best = int(boxes.conf.argmax())
        return (boxes.xyxy[best].cpu().numpy().astype(np.float32),
                float(boxes.conf[best]), ms)


def install_flash_attn_stub() -> None:
    """向 sys.modules 注入 flash_attn 存根，绕过 transformers check_imports 的硬性扫描。

    Windows 无 flash-attn wheel，而 Florence-2 远程 modeling 文件的
    `if is_flash_attn_2_available(): from flash_attn import ...` 块会被
    check_imports 当成无条件依赖。存根满足 importlib.import_module 的检查；
    is_flash_attn_2_available() 走 importlib.metadata（发行版元数据），
    存根不会使其变 True —— 即真实 flash-attn 路径永远不会被实例化，
    模型实际走 SDPA 注意力，安全。
    """
    import types
    import importlib.machinery
    if "flash_attn" in sys.modules:
        return

    def _forbidden(*args, **kwargs):
        raise RuntimeError("flash_attn stub: real flash-attention was never meant to run here")

    def _make_module(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__path__ = []  # 标记为包，允许子模块导入
        mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
        mod.__loader__ = None
        return mod

    fa = _make_module("flash_attn")
    bp = _make_module("flash_attn.bert_padding")
    for fn_name in ("index_first_axis", "pad_input", "unpad_input"):
        setattr(bp, fn_name, _forbidden)
    fa.bert_padding = bp
    sys.modules["flash_attn"] = fa
    sys.modules["flash_attn.bert_padding"] = bp


class FlorenceAdapter:
    """Florence-2 惰性适配器：import/权重下载任一步失败即抛异常，由上层降级。"""
    name = "florence2"

    def __init__(self, model_id: str = "microsoft/Florence-2-base", device: Optional[str] = None):
        install_flash_attn_stub()
        from transformers import AutoProcessor, AutoModelForCausalLM
        self.model_id = model_id
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"⏳ 加载 Florence-2: {model_id} ({self.device}) — 首次运行需下载权重")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True
        ).to(self.device).eval()
        # 预热
        dummy = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
        self._generate(dummy, "<OD>", "")

    # 纯任务型 token（如 <OD>）：text 必须为空；with-input 型（如接地）：text 为短语
    PURE_TASKS = {"<OD>", "<DENSE_REGION_CAPTION>", "<REGION_PROPOSAL>"}

    def _generate(self, image: Image.Image, task: str, text: str) -> dict:
        if task in self.PURE_TASKS:
            prompt_text = task
        else:
            prompt_text = f"{task}{text}" if text else task
        inputs = self.processor(images=image, text=prompt_text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if "pixel_values" in inputs and inputs["pixel_values"].dtype == torch.float32 \
                and self.device == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=1024, num_beams=3, early_stopping=True
            )
        # 此版本签名是 (text, sequence, ..., task, image_size)，
        # 且 image_size 实际按 (W, H) 解释（与 docstring 相反，实测 y 超出 1080 所致）
        out = self.processor.post_process_generation(
            text=None, sequence=generated[0], task=task, image_size=image.size
        )
        # 结果嵌套在任务键下：{task: {"bboxes": ..., "labels": ...}}
        return out.get(task) or {}

    def detect(self, image: Image.Image, prompt: str):
        t0 = time.perf_counter()
        box, conf = None, None
        # 1) 短语接地检测（与原 FlorenceBoxExtractor 语义最接近）
        for text in (prompt, f"a {prompt}"):
            try:
                out = self._generate(image, "<CAPTION_TO_PHRASE_GROUNDING>", text)
                bboxes = out.get("bboxes") or []
                if bboxes:
                    box = np.asarray(bboxes[0], dtype=np.float32)
                    conf = float("nan")  # Florence-2 不输出置信度
                    break
            except Exception as e:  # 单任务失败不致命
                print(f"⚠️  Florence-2 <CAPTION_TO_PHRASE_GROUNDING> 失败: {e}")
        # 2) 开放检测兜底：按标签匹配目标名
        if box is None:
            try:
                out = self._generate(image, "<OD>", "")
                for lbl, bb in zip(out.get("labels") or [], out.get("bboxes") or []):
                    if label_matches_prompt(lbl, prompt):
                        box = np.asarray(bb, dtype=np.float32)
                        conf = float("nan")
                        break
            except Exception as e:
                print(f"⚠️  Florence-2 <OD> 失败: {e}")
        ms = (time.perf_counter() - t0) * 1000
        if box is None:
            print(f"⚠️  Florence-2 未检出 '{prompt}'")
        return box, conf, ms


# ----------------------------------------------------------------------------
# 实验层
# ----------------------------------------------------------------------------

def load_first_frame(target: str) -> Tuple[Image.Image, Path]:
    base, _, _, frame_dir, _, _ = build_project_paths()
    frame_dir = frame_dir.parent / target
    frames = sorted([p for p in frame_dir.iterdir() if p.suffix.lower() in (".jpg", ".png")])
    if not frames:
        raise FileNotFoundError(f"{frame_dir} 无帧图片")
    return Image.open(frames[0]).convert("RGB"), frames


def run_box_layer(target: str, image: Image.Image, detectors: List) -> Tuple[dict, List]:
    """层1：各检测器在首帧上的 box / 置信度 / 时延 + 交叉 IoU。返回 (row, overlay_boxes)。"""
    row = {"target": target}
    boxes = {}
    overlay_boxes = []
    for det in detectors:
        try:
            box, conf, ms = det.detect(image, target)
            boxes[det.name] = box
            row[f"box_{det.name}"] = None if box is None else box.tolist()
            row[f"conf_{det.name}"] = conf
            row[f"latency_ms_{det.name}"] = round(ms, 1)
            if box is not None:
                overlay_boxes.append((det.name, box))
            print(f"   [{det.name}] box={None if box is None else box.round().tolist()}, "
                  f"conf={conf}, {ms:.0f}ms")
        except Exception as e:
            print(f"   [{det.name}] 检测失败: {e}")
            row[f"error_{det.name}"] = str(e)
    names = [d.name for d in detectors]
    if len(names) >= 2 and boxes.get(names[0]) is not None and boxes.get(names[1]) is not None:
        row["cross_iou"] = round(box_iou(boxes[names[0]], boxes[names[1]]), 4)
        print(f"   交叉 IoU({names[0]} vs {names[1]}) = {row['cross_iou']}")
    return row, overlay_boxes


def run_mask_layer(target: str, frame_paths: List[Path], detectors: List,
                   max_frames: int) -> List[dict]:
    """层2：每个检测器的 box 各跑一次 SAM2 传播（帧子集），交叉 IoU + 时序稳定性。

    mask 缓存写入 caches/mask/{target}_{detector}.pt，与主管线缓存隔离。
    """
    from reasoning.mask_predictor import MaskExtractor
    extractor = MaskExtractor()
    subset = frame_paths[:max_frames]
    masks_by_det = {}
    rows = []
    for det in detectors:
        det_target = f"{target}_{det.name}"
        try:
            image0 = Image.open(subset[0]).convert("RGB")
            box, _, _ = det.detect(image0, target)
            if box is None:
                print(f"   [{det.name}] 无 box，跳过 mask 传播")
                continue
            print(f"   [{det.name}] SAM2 传播 {len(subset)} 帧 ...")
            masks = extractor.extract_masks_from_video(
                frame_paths=subset,
                initial_box=np.asarray(box, dtype=np.float32),
                target_name=det_target,      # 独立缓存名，不碰 caches/mask/{target}.pt
                save_visualization=False,
                use_cache=True,
            )
            masks_by_det[det.name] = masks
            stability = float(np.mean([
                mask_iou(masks[i], masks[i + 1]) for i in range(len(masks) - 1)
            ]))
            coverage = float(np.mean(masks))
            rows.append({"target": target, "detector": det.name,
                         "mask_temporal_iou": round(stability, 4),
                         "mask_mean_coverage": round(coverage, 4)})
            print(f"   [{det.name}] 时序稳定 IoU={stability:.3f}, 平均覆盖={coverage:.3f}")
        except Exception:
            print(f"   [{det.name}] mask 层失败:\n{traceback.format_exc(limit=3)}")
    # 交叉 IoU：两检测器 mask 序列逐帧对比
    names = [d.name for d in detectors]
    if len(names) >= 2 and names[0] in masks_by_det and names[1] in masks_by_det:
        m1, m2 = masks_by_det[names[0]], masks_by_det[names[1]]
        n = min(len(m1), len(m2))
        cross = float(np.mean([mask_iou(m1[i], m2[i]) for i in range(n)]))
        rows.append({"target": target, "detector": f"{names[0]}_vs_{names[1]}",
                     "mask_cross_iou": round(cross, 4)})
        print(f"   mask 序列交叉 IoU = {cross:.3f}")
    return rows


def save_box_overlay(target: str, image: Image.Image, boxes: List[Tuple[str, np.ndarray]],
                     out_dir: Path):
    """首帧叠加两检测器 box（YOLO 绿 / Florence 红），直观对比。"""
    vis = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    colors = {"yolo11m": (0, 255, 0), "florence2": (0, 0, 255)}
    for name, box in boxes:
        if box is None:
            continue
        x1, y1, x2, y2 = box.round().astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), colors.get(name, (255, 255, 0)), 3)
        cv2.putText(vis, name, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors.get(name, (255, 255, 0)), 2)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{target}_frame0_boxes.png"), vis)


def flatten_rows(rows: List[dict]) -> List[dict]:
    """把嵌套 dict 拍平成长表行，方便统一写 CSV。"""
    flat = []
    for r in rows:
        base = {"target": r.get("target")}
        for k, v in r.items():
            if k in ("target", "detector"):
                continue
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    flat.append({**base, "metric": k2, "value": str(v2), "detector": "-"})
            else:
                flat.append({**base, "metric": k, "value": v, "detector": r.get("detector", "-")})
    return flat


def main():
    parser = argparse.ArgumentParser(description="GT-free 双检测器对比（YOLO vs Florence-2）")
    parser.add_argument("--targets", nargs="+", default=["bear"],
                        help="目标名列表，如 bear blackswan car cows")
    parser.add_argument("--max_frames", type=int, default=20,
                        help="层2 SAM2 传播的帧数上限（控制时长）")
    parser.add_argument("--skip_mask", action="store_true", help="只跑层1 box 对比")
    parser.add_argument("--no_florence", action="store_true", help="禁用 Florence-2，YOLO 单检测器模式")
    parser.add_argument("--florence_model", type=str, default="microsoft/Florence-2-base")
    parser.add_argument("--yolo_weights", type=str, default="yolo11m.pt")
    args = parser.parse_args()

    base, _, _, _, result_dir, _ = build_project_paths()
    out_dir = result_dir / "detector_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 构建检测器（每个失败都降级而不是退出）---
    detectors = []
    try:
        detectors.append(YoloAdapter(weights=args.yolo_weights))
    except Exception:
        print(f"❌ YOLO 初始化失败，实验无法进行:\n{traceback.format_exc(limit=5)}")
        return 1

    if not args.no_florence:
        try:
            detectors.append(FlorenceAdapter(model_id=args.florence_model))
            print("✅ Florence-2 可用，进行双检测器对比")
        except Exception as e:
            print(f"⚠️  Florence-2 不可用（{e}），降级为 YOLO 单检测器模式，层1/2 交叉指标将缺失")

    all_rows = []
    for target in args.targets:
        print(f"\n{'=' * 60}\n🎯 目标: {target}")
        try:
            image, frame_paths = load_first_frame(target)
        except Exception as e:
            print(f"⚠️  跳过 {target}: {e}")
            continue

        # 层1
        try:
            row1, overlay_boxes = run_box_layer(target, image, detectors)
            all_rows.append(row1)
            if overlay_boxes:
                save_box_overlay(target, image, overlay_boxes, out_dir)
        except Exception:
            print(f"⚠️  层1 失败:\n{traceback.format_exc(limit=3)}")

        # 层2
        if not args.skip_mask:
            try:
                all_rows.extend(run_mask_layer(target, frame_paths, detectors, args.max_frames))
            except Exception:
                print(f"⚠️  层2 失败:\n{traceback.format_exc(limit=3)}")

    # --- 输出 CSV ---
    csv_path = out_dir / "metrics.csv"
    flat = flatten_rows(all_rows)
    if flat:
        fieldnames = ["target", "detector", "metric", "value"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat)
        print(f"\n💾 指标已写入 {csv_path}")

    # --- 汇总报告 ---
    print(f"\n{'=' * 60}\n📊 汇总 ({len(detectors)} 个检测器)")
    for r in all_rows:
        if "cross_iou" in r:
            print(f"  {r['target']}: box 交叉 IoU = {r['cross_iou']}")
        for k in ("mask_temporal_iou", "mask_cross_iou", "mask_mean_coverage"):
            if k in r:
                print(f"  {r['target']} [{r.get('detector', '-')}]: {k} = {r[k]}")
    print(f"🖼️  叠加图: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
