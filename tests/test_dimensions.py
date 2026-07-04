"""dimensions 模块的基础单元测试。用纯色图验证维度计算方向正确。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from aesthetic.dimensions import extract_dimensions  # noqa: E402


def _solid(color):
    return Image.fromarray(np.full((64, 64, 3), color, dtype=np.uint8))


def test_black_is_dark():
    d = extract_dimensions(_solid((0, 0, 0)))
    assert d.lightness < 0.1, "纯黑应判为极暗"


def test_white_is_bright():
    d = extract_dimensions(_solid((255, 255, 255)))
    assert d.lightness > 0.9, "纯白应判为极亮"


def test_red_is_warm():
    d = extract_dimensions(_solid((200, 30, 30)))
    assert d.warmth > 0.55, "红色应判为暖调"


def test_blue_is_cold():
    d = extract_dimensions(_solid((30, 30, 200)))
    assert d.warmth < 0.45, "蓝色应判为冷调"


def test_gray_is_low_saturation():
    d = extract_dimensions(_solid((128, 128, 128)))
    assert d.saturation < 0.1, "灰色应判为低饱和"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
    print("全部通过")
