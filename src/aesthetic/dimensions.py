"""
第一层:客观视觉维度提取
------------------------
从像素直接计算 5 个跨题材通用的客观维度,不需要任何模型或标注:
  明暗 / 饱和度 / 冷暖 / 对比 / 色彩丰富度

每个维度是 0~1 的连续值,并提供离散档位标签。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

# 档位分界线(可按数据分布校准)
_LIGHT_DARK, _LIGHT_BRIGHT = 0.40, 0.60
_SAT_LOW, _SAT_HIGH = 0.35, 0.60
_WARM_COLD, _WARM_WARM = 0.45, 0.55
_CONTRAST_STRONG = 0.55
_COLORFUL_RICH = 0.45  # 从0.5下调到0.45,让边界图分类更合理


@dataclass
class VisualDimensions:
    lightness: float
    saturation: float
    warmth: float
    contrast: float
    colorfulness: float

    def label_lightness(self) -> str:
        return "暗调" if self.lightness < _LIGHT_DARK else ("明亮" if self.lightness > _LIGHT_BRIGHT else "中间调")

    def label_saturation(self) -> str:
        return "低饱和" if self.saturation < _SAT_LOW else ("高饱和" if self.saturation > _SAT_HIGH else "中饱和")

    def label_warmth(self) -> str:
        return "冷调" if self.warmth < _WARM_COLD else ("暖调" if self.warmth > _WARM_WARM else "中性")

    def label_contrast(self) -> str:
        return "强对比" if self.contrast > _CONTRAST_STRONG else "柔和"

    def label_colorfulness(self) -> str:
        return "丰富" if self.colorfulness > _COLORFUL_RICH else "极简"


def extract_dimensions(img: Image.Image) -> VisualDimensions:
    """从一张 PIL 图片计算 5 个客观维度。"""
    small = img.resize((128, 128)).convert("RGB")
    arr = np.asarray(small).astype(float) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    lightness = float(luminance.mean())

    mx, mn = arr.max(axis=2), arr.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / (mx + 1e-6), 0)
    saturation = float(sat.mean())

    warmth = float(np.clip((r.mean() - b.mean()) * 2 + 0.5, 0, 1))
    contrast = float(np.clip(luminance.std() * 3.0, 0, 1))

    hsv = np.array(Image.fromarray((arr * 255).astype(np.uint8)).convert("HSV")).astype(float)
    hue = hsv[..., 0] / 255.0
    sat_mask = sat > 0.2
    if sat_mask.sum() > 0:
        hue_bins = np.histogram(hue[sat_mask], bins=12, range=(0, 1))[0]
        active_hues = int((hue_bins > hue_bins.sum() * 0.05).sum())
        colorfulness = float(np.clip(active_hues / 8.0, 0, 1))
    else:
        colorfulness = 0.0

    return VisualDimensions(lightness, saturation, warmth, contrast, colorfulness)
