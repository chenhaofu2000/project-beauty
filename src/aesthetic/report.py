"""
分析编排 + 报告生成
--------------------
把三条能力(题材/维度/情绪)编排起来,对一批图片生成审美画像报告。

- analyze_folder: 分析一个文件夹,返回结构化结果
- format_report: 把结构化结果渲染成文字报告
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter

import numpy as np
from PIL import Image
from loguru import logger

from .dimensions import VisualDimensions, extract_dimensions
from .mood import map_to_mood
from .subject import SubjectClassifier

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class ImageAnalysis:
    path: str
    dims: VisualDimensions
    mood: str
    subject: str
    subject_conf: float


@dataclass
class AestheticProfile:
    """整批图片的审美画像(结构化结果,便于下游:渲染/LLM/前端)。"""
    analyses: list[ImageAnalysis]
    subject_dist: dict[str, float]      # 题材 -> 占比
    mood_dist: dict[str, float]         # 情绪 -> 占比
    avg_lightness: float
    avg_saturation: float
    avg_warmth: float


def analyze_folder(folder: Path) -> "AestheticProfile | None":
    classifier = SubjectClassifier()
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXT)

    analyses: list[ImageAnalysis] = []
    for path in paths:
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            logger.warning(f"跳过 {path.name}: {e}")
            continue
        dims = extract_dimensions(img)
        subject, conf = classifier.classify(img)
        analyses.append(ImageAnalysis(str(path), dims, map_to_mood(dims), subject, conf))

    if not analyses:
        logger.error("没有可分析的图片")
        return None

    n = len(analyses)
    subj_counts = Counter(a.subject for a in analyses)
    mood_counts = Counter(a.mood for a in analyses)

    return AestheticProfile(
        analyses=analyses,
        subject_dist={k: v / n for k, v in subj_counts.most_common()},
        mood_dist={k: v / n for k, v in mood_counts.most_common()},
        avg_lightness=float(np.mean([a.dims.lightness for a in analyses])),
        avg_saturation=float(np.mean([a.dims.saturation for a in analyses])),
        avg_warmth=float(np.mean([a.dims.warmth for a in analyses])),
    )


def format_report(profile: AestheticProfile) -> str:
    lines: list[str] = []

    lines.append("=== 每张图:数值 + 标签 + 情绪 + 题材 ===")
    for a in profile.analyses:
        d = a.dims
        lines.append(
            f"  {Path(a.path).name[:15].ljust(16)} "
            f"明{d.lightness:.2f} 饱{d.saturation:.2f} 暖{d.warmth:.2f} "
            f"对{d.contrast:.2f} 彩{d.colorfulness:.2f}  "
            f"{a.mood} | {a.subject}({a.subject_conf:.2f})"
        )

    lines.append("\n=== 题材分布 ===")
    for subj, pct in profile.subject_dist.items():
        lines.append(f"  {subj.ljust(6)} {round(pct*100):3d}%  {'#' * int(pct*20)}")

    lines.append("\n=== 情绪分布 ===")
    mood_parts = []
    for mood, pct in profile.mood_dist.items():
        mood_parts.append(f"{round(pct*100)}% {mood}")
        lines.append(f"  {mood.ljust(6)} {round(pct*100):3d}%  {'#' * int(pct*20)}")

    # 整体调性
    tone = []
    tone.append("偏暗" if profile.avg_lightness < 0.45 else ("偏亮" if profile.avg_lightness > 0.6 else "明暗均衡"))
    tone.append("低饱和高级感" if profile.avg_saturation < 0.38 else ("色彩浓郁" if profile.avg_saturation > 0.6 else "饱和适中"))
    tone.append("冷调" if profile.avg_warmth < 0.45 else ("暖调" if profile.avg_warmth > 0.55 else "冷暖平衡"))

    top_subj = next(iter(profile.subject_dist.items()))
    lines.append("\n── 报告雏形 ──")
    lines.append(f"  你偏爱【{top_subj[0]}】题材({round(top_subj[1]*100)}%),")
    lines.append(f"  审美由 {' + '.join(mood_parts[:3])} 构成,")
    lines.append(f"  整体调性:{' / '.join(tone)}。")

    return "\n".join(lines)
