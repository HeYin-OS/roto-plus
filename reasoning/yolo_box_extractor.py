"""基于 YOLO 的提示框提取类。

接口与 FlorenceBoxExtractor 完全对齐（extract_boxes(image, text_prompt)），
作为 mask_predictor 流水线中「文本 -> 目标框」环节的 YOLO 实现。

使用 COCO 预训练权重（yolo11m），目标名通过同义词表映射到 COCO 类别：
  bear  -> bear
  cows  -> cow
  blackswan -> bird
  car   -> car
"""
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from ultralytics import YOLO

# 项目目标名 -> COCO 类名 的同义词映射（小写，单数）
COCO_SYNONYMS = {
    "cows": "cow",
    "blackswan": "bird",
    "swan": "bird",
    "bmx": None,      # 非物体类别（骑手场景），无法用 COCO 检出
    "boat": "boat",
    "bus": "bus",
    "frog": None,
    "parachute": None,
    "ball": "sports ball",
    "car": "car",
    "bear": "bear",
}

# 默认权重：m 档兼顾精度与速度，RTX 4070 上首帧 ~几十 ms
DEFAULT_WEIGHTS = "yolo11m.pt"
DEFAULT_CONF = 0.25


class YoloBoxExtractor:
    """基于 YOLO 的提示框提取类（替代 FlorenceBoxExtractor）"""

    def __init__(self, weights: str = DEFAULT_WEIGHTS, device: Optional[str] = None,
                 conf: float = DEFAULT_CONF):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ 初始化 YOLO ({weights}, {self.device})")
        # 首次使用自动下载 COCO 预训练权重
        self.model = YOLO(weights)
        self.conf = conf
        # 模型类名表（小写），用于目标名兜底匹配
        self._coco_names = {name.lower() for name in self.model.names.values()}

    def resolve_coco_class(self, text_prompt: str) -> Optional[str]:
        """将项目目标名解析为 COCO 类名；无法映射时返回 None。"""
        prompt = text_prompt.strip().lower()
        # 精确同义词表优先
        if prompt in COCO_SYNONYMS:
            return COCO_SYNONYMS[prompt]
        # 简单复数兜底：cows -> cow
        if prompt.endswith("s") and prompt[:-1] in COCO_SYNONYMS:
            return COCO_SYNONYMS[prompt[:-1]]
        # 本身就是 COCO 类名（如 "car", "bear"）
        if prompt in self._coco_names:
            return prompt
        return None

    def extract_boxes(self, image: Image.Image, text_prompt: str) -> List[np.ndarray]:
        """检测图中目标，返回 [[x1, y1, x2, y2], ...] 列表（与 Florence-2 输出格式一致）。

        无匹配类别或未检出目标时返回空列表。
        """
        coco_class = self.resolve_coco_class(text_prompt)
        if coco_class is None:
            print(f"⚠️  目标 '{text_prompt}' 无法映射到 COCO 类别，跳过检测")
            return []

        class_ids = [cid for cid, name in self.model.names.items()
                     if name.lower() == coco_class]
        if not class_ids:
            print(f"⚠️  COCO 类别 '{coco_class}' 不在模型类别表中")
            return []

        results = self.model.predict(
            source=np.array(image),
            conf=self.conf,
            classes=class_ids,
            device=self.device,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return []

        # 取置信度最高的检测结果
        boxes = results[0].boxes
        if len(boxes) == 0:
            print(f"⚠️  YOLO 未检测到 '{coco_class}'")
            return []

        best = int(boxes.conf.argmax())
        xyxy = boxes.xyxy[best].cpu().numpy().astype(np.float32)
        print(f"✅ YOLO 检出 '{coco_class}' (conf={float(boxes.conf[best]):.2f}, box={xyxy.tolist()})")
        return [xyxy]
