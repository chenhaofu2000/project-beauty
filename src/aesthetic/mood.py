"""
第二层:情绪映射
----------------
把第一层的客观维度组合成情绪标签(规则映射,不需训练)。

注:规则表是"知识",可按个人审美调整。当前版本区分了
"暖调的暗"(温郁/沉静)和"冷调的暗"(忧郁),避免把偏暖的暗调
误判为普通"温暖"。
"""

from __future__ import annotations

from .dimensions import VisualDimensions


def map_to_mood(d: VisualDimensions) -> str:
    dark = d.lightness < 0.42
    bright = d.lightness > 0.60
    low_sat = d.saturation < 0.35
    high_sat = d.saturation > 0.60
    cold = d.warmth < 0.45
    warm = d.warmth > 0.55
    soft = d.contrast < 0.50

    # 暗调系(先细分冷暖)
    if dark and cold and low_sat:
        return "忧郁"
    if dark and warm and low_sat:
        return "温郁"          # 暖调的暗:复古/沉静/有温度的暗
    if dark and cold:
        return "静谧"
    if dark and high_sat:
        return "浓烈"
    if dark:
        return "沉静"

    # 明亮系
    if bright and warm and not low_sat:
        return "明媚"
    if bright and soft and low_sat:
        return "梦幻"
    if bright and warm and soft:
        return "温柔"

    # 中间调 / 其他
    if low_sat and cold:
        return "清冷"
    if high_sat:
        return "鲜活"
    if warm:
        return "温暖"
    if cold:
        return "冷峻"
    return "平和"
