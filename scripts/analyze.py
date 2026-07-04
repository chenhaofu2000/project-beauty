"""
命令行入口:分析一个图片文件夹,输出审美报告。

用法:
    python scripts/analyze.py <图片文件夹>
"""
import sys
from pathlib import Path

# 让脚本能找到 src/ 下的包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aesthetic import analyze_folder, format_report  # noqa: E402
from loguru import logger  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python scripts/analyze.py <图片文件夹>")
        sys.exit(1)
    folder = Path(sys.argv[1])
    if not folder.is_dir():
        logger.error(f"{folder} 不是有效文件夹")
        sys.exit(1)

    profile = analyze_folder(folder)
    if profile is None:
        return
    logger.success(f"分析完成:{len(profile.analyses)} 张图\n")
    print(format_report(profile))


if __name__ == "__main__":
    main()
