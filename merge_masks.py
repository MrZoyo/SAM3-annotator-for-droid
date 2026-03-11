"""将每帧 5 种 mask PNG 合并为 .npz 文件。

对于每个相机的每一帧，读取 pick/place/push/pull/press 五个目录下的 mask PNG，
合并为一个 .npz，每个键对应一种操作类型，值为 (H, W) 的 uint8 数组 (0/1)。

用法:
    python merge_masks.py /data1/yaoxuran/press_one_button_demo/episode_00000

输入目录结构:
    {episode}/mask/{camera}/left/pick/00000.png ...

输出:
    {episode}/heatmap/{camera}/left/00000.npz
    读取: data = np.load("00000.npz"); data["pick"]  # (H, W), 0/1
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

MASK_TYPES = ["pick", "place", "push", "pull", "press"]


def merge_episode(episode_path: str):
    ep = Path(episode_path)
    mask_root = ep / "mask"
    if not mask_root.is_dir():
        print(f"未找到 mask 目录: {mask_root}")
        return

    # 检测相机
    cameras = sorted(
        d.name for d in mask_root.iterdir()
        if d.is_dir() and (d / "left").is_dir()
    )
    if not cameras:
        print(f"未找到任何相机目录: {mask_root}")
        return

    for cam in cameras:
        left_dir = mask_root / cam / "left"
        out_dir = ep / "heatmap" / cam / "left"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 收集所有帧文件名（取所有类型的并集）
        all_frames: set[str] = set()
        for mt in MASK_TYPES:
            mt_dir = left_dir / mt
            if mt_dir.is_dir():
                all_frames.update(p.stem for p in mt_dir.glob("*.png"))

        if not all_frames:
            print(f"  {cam}: 无 mask 文件，跳过")
            continue

        sorted_frames = sorted(all_frames)
        count = 0

        for frame_name in sorted_frames:
            masks = {}
            for mt in MASK_TYPES:
                png_path = left_dir / mt / f"{frame_name}.png"
                if png_path.exists():
                    img = np.array(Image.open(png_path).convert("L"))
                    masks[mt] = (img > 127).astype(np.uint8)
                else:
                    masks[mt] = None

            # 获取尺寸（从已有 mask 中取）
            h, w = 0, 0
            for m in masks.values():
                if m is not None:
                    h, w = m.shape
                    break

            # 补全缺失的 mask 为全零
            for mt in MASK_TYPES:
                if masks[mt] is None:
                    masks[mt] = np.zeros((h, w), dtype=np.uint8)

            np.savez(out_dir / f"{frame_name}.npz", **masks)
            count += 1

        print(f"  {cam}: {count} 帧 → {out_dir}")

    print("完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并 5 种 mask PNG 为 .npz")
    parser.add_argument("episode_path", help="Episode 目录路径")
    args = parser.parse_args()
    merge_episode(args.episode_path)
