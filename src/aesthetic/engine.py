"""
CLIP 引擎(单例)
----------------
集中管理 CLIP 模型的加载与基础编码能力。
以前 CLIP 加载逻辑散落在多个脚本里,这里统一成一处,处处复用。

- 图像编码(用于题材分类等)
- 文本锚点编码(用于把一组提示词编码成"锚点向量")
"""

from __future__ import annotations

import numpy as np
import torch
import open_clip
from PIL import Image
from loguru import logger

_MODEL_NAME = "ViT-B-32"
_PRETRAINED = "laion2b_s34b_b79k"


class ClipEngine:
    """CLIP 模型封装。加载昂贵,建议全局只实例化一次。"""

    _instance: "ClipEngine | None" = None

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"加载 CLIP {_MODEL_NAME}/{_PRETRAINED} 到 {self.device} ...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            _MODEL_NAME, pretrained=_PRETRAINED
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(_MODEL_NAME)
        logger.success("CLIP 加载完成")

    @classmethod
    def instance(cls) -> "ClipEngine":
        """获取全局单例,避免重复加载模型。"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def encode_image(self, img: Image.Image) -> np.ndarray:
        """一张图 -> L2 归一化的 512 维向量。"""
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(tensor)
            feat /= feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().flatten()

    def encode_text_anchors(self, prompt_groups: dict[str, list[str]]) -> tuple[list[str], np.ndarray]:
        """
        把每组提示词编码成一个"锚点向量"(同组多个提示词取平均)。
        返回 (名称列表, 锚点矩阵[N,512])。
        用于题材/情绪等基于文本提示的分类。
        """
        names = list(prompt_groups.keys())
        anchors = []
        for name in names:
            tokens = self.tokenizer(prompt_groups[name]).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_text(tokens)
                feats /= feats.norm(dim=-1, keepdim=True)
                mean_feat = feats.mean(dim=0)
                mean_feat /= mean_feat.norm()
            anchors.append(mean_feat.cpu().numpy())
        return names, np.stack(anchors)
