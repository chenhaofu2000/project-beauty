"""
CLIP 题材分类腿
----------------
用一组题材提示词,通过 CLIP 判断每张图的题材
(人像/风景/卡通/插画/建筑/静物/食物/动物)。

这是 CLIP 最擅长的任务(语义/内容识别),准确率高。
题材是普通用户审美的重要信号(很多人偏好集中在某类题材)。
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .engine import ClipEngine

# 题材词表。以后可扩展为更灵活的方式(自定义/层级题材)。
SUBJECT_PROMPTS: dict[str, list[str]] = {
    "人像": ["a portrait of a person", "a photo of a person", "a selfie"],
    "风景": ["a landscape photo", "a scenery view", "nature scenery"],
    "卡通": ["a cartoon", "an anime illustration", "a cartoon character"],
    "插画": ["a digital illustration", "an artistic painting", "concept art"],
    "建筑": ["a building", "architecture photo", "a cityscape"],
    "静物": ["a still life photo", "an object close-up", "a product photo"],
    "食物": ["a photo of food", "a dish of food", "a meal"],
    "动物": ["a photo of an animal", "a pet photo", "a cute animal"],
}


class SubjectClassifier:
    def __init__(self, engine: "ClipEngine | None" = None) -> None:
        self.engine = engine or ClipEngine.instance()
        self.names, self.anchors = self.engine.encode_text_anchors(SUBJECT_PROMPTS)

    def classify(self, img: Image.Image) -> tuple[str, float]:
        """返回 (题材名, 置信度)。置信度是与最佳题材锚点的余弦相似度。"""
        vec = self.engine.encode_image(img)
        sims = self.anchors @ vec
        idx = int(sims.argmax())
        return self.names[idx], float(sims[idx])
