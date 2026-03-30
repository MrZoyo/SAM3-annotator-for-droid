"""数据集读取、帧加载、lang.txt 读写、mask 保存工具。"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

MASK_TYPES = ["pick", "place", "push", "pull", "press"]

MASK_TYPE_CN = {
    "pick": "抓取目标",
    "place": "放置目标",
    "push": "推目标",
    "pull": "拉目标",
    "press": "按目标",
}


def detect_format(episode_path: str) -> str:
    """检测 episode 的数据集格式。

    返回 "stereo"（DROID 立体双目）或 "mono"（单目）。
    stereo: rgb_stereo_valid/{camera}/left/*.png
    mono:   rgb/{camera}/*.png
    """
    ep = Path(episode_path)
    if (ep / "rgb_stereo_valid").is_dir():
        return "stereo"
    if (ep / "rgb").is_dir():
        return "mono"
    return "unknown"


def list_cameras(episode_path: str) -> list[str]:
    """自动检测 episode 下的相机目录列表。"""
    fmt = detect_format(episode_path)
    if fmt == "stereo":
        stereo_dir = Path(episode_path) / "rgb_stereo_valid"
        return sorted(
            d.name
            for d in stereo_dir.iterdir()
            if d.is_dir() and (d / "left").is_dir()
        )
    elif fmt == "mono":
        rgb_dir = Path(episode_path) / "rgb"
        return sorted(
            d.name
            for d in rgb_dir.iterdir()
            if d.is_dir() and any(d.glob("*.png"))
        )
    return []


def list_episodes(dataset_root: str) -> list[str]:
    """列出数据集根目录下所有 episode 目录名。"""
    root = Path(dataset_root)
    if not root.is_dir():
        return []
    episodes = sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir() and (
            (d / "rgb_stereo_valid").is_dir() or (d / "rgb").is_dir()
        )
    )
    return episodes


def get_frames_dir(episode_path: str, camera: str) -> Path:
    """获取指定 episode + camera 的帧目录。自动适配 stereo/mono 格式。"""
    fmt = detect_format(episode_path)
    if fmt == "stereo":
        return Path(episode_path) / "rgb_stereo_valid" / camera / "left"
    else:
        return Path(episode_path) / "rgb" / camera


def load_frame_paths(episode_path: str, camera: str) -> list[Path]:
    """加载指定视角的所有帧路径，按名称排序。"""
    frames_dir = get_frames_dir(episode_path, camera)
    if not frames_dir.is_dir():
        return []
    return sorted(frames_dir.glob("*.png"))


def load_frame(episode_path: str, camera: str, frame_idx: int) -> np.ndarray | None:
    """加载单帧图像为 numpy 数组 (H, W, 3)。"""
    paths = load_frame_paths(episode_path, camera)
    if frame_idx < 0 or frame_idx >= len(paths):
        return None
    return np.array(Image.open(paths[frame_idx]).convert("RGB"))


def get_frame_count(episode_path: str, camera: str) -> int:
    """获取帧数量。"""
    return len(load_frame_paths(episode_path, camera))


def read_lang(episode_path: str) -> str:
    """读取 episode 下的 lang.txt。"""
    lang_file = Path(episode_path) / "lang.txt"
    if lang_file.exists():
        return lang_file.read_text(encoding="utf-8").strip()
    return ""


def save_lang(episode_path: str, text: str):
    """保存 lang.txt。"""
    lang_file = Path(episode_path) / "lang.txt"
    lang_file.write_text(text.strip() + "\n", encoding="utf-8")


def _mask_dir(episode_path: str, camera: str, mask_type: str) -> Path:
    """获取 mask 保存目录，自动适配格式。"""
    fmt = detect_format(episode_path)
    if fmt == "stereo":
        return Path(episode_path) / "mask" / camera / "left" / mask_type
    else:
        return Path(episode_path) / "mask" / camera / mask_type


def save_masks(
    episode_path: str,
    camera: str,
    mask_type: str,
    masks: dict[int, np.ndarray],
    total_frames: int,
) -> str:
    """保存完整序列的逐帧二值 mask。

    masks: {frame_idx: bool array (H, W)}
    total_frames: 视频总帧数，确保输出完整序列
    保存为 0/255 的 PNG。没有 mask 的帧保存全零。
    返回保存目录路径。
    """
    out_dir = _mask_dir(episode_path, camera, mask_type)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 获取 mask 尺寸
    if masks:
        sample = next(iter(masks.values()))
        h, w = sample.shape[:2]
    else:
        h, w = 480, 640

    for fidx in range(total_frames):
        if fidx in masks:
            mask_img = Image.fromarray((masks[fidx].astype(np.uint8) * 255))
        else:
            mask_img = Image.fromarray(np.zeros((h, w), dtype=np.uint8))
        mask_img.save(out_dir / f"{fidx:05d}.png")

    return str(out_dir)


INVALID_MARK_FILE = "INVALID_MARK"


def is_episode_valid(episode_path: str) -> bool:
    """检查 episode 是否可用（不存在 INVALID_MARK 文件）。"""
    return not (Path(episode_path) / INVALID_MARK_FILE).exists()


def set_episode_valid(episode_path: str, valid: bool):
    """设置 episode 可用/不可用状态。"""
    mark_file = Path(episode_path) / INVALID_MARK_FILE
    if valid:
        if mark_file.exists():
            mark_file.unlink()
    else:
        mark_file.touch()


def delete_saved_masks(episode_path: str, camera: str) -> list[str]:
    """删除指定视角的所有已保存 mask。返回被删除的类型列表。"""
    import shutil

    deleted = []
    for mt in MASK_TYPES:
        mask_dir = _mask_dir(episode_path, camera, mt)
        if mask_dir.is_dir() and any(mask_dir.glob("*.png")):
            shutil.rmtree(mask_dir)
            deleted.append(mt)
    return deleted


def get_saved_status(episode_path: str) -> dict[str, dict[str, bool]]:
    """检查已保存的 mask 状态。返回 {camera: {mask_type: bool}}。"""
    cameras = list_cameras(episode_path)
    status = {}
    for cam in cameras:
        status[cam] = {}
        for mt in MASK_TYPES:
            mask_dir = _mask_dir(episode_path, cam, mt)
            status[cam][mt] = mask_dir.is_dir() and any(mask_dir.glob("*.png"))
    return status


_video_cache: dict[str, str] = {}  # (ep_path, camera) key → mp4 path


def build_preview_video(episode_path: str, camera: str, fps: int = 15) -> str | None:
    """用 ffmpeg 将 PNG 帧序列合成临时 mp4，返回视频文件路径。结果会缓存。"""
    cache_key = f"{episode_path}|{camera}"
    if cache_key in _video_cache:
        cached = _video_cache[cache_key]
        if Path(cached).exists():
            return cached

    frames_dir = get_frames_dir(episode_path, camera)
    if not frames_dir.is_dir():
        return None

    out_path = tempfile.mktemp(suffix=".mp4", prefix=f"preview_{camera}_")
    pattern = str(frames_dir / "%05d.png")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        "-preset", "fast",
        "-loglevel", "error",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None

    _video_cache[cache_key] = out_path
    return out_path


def build_result_video(
    episode_path: str,
    camera: str,
    tracked_masks: dict[str, dict[int, np.ndarray]],
    mask_colors: dict[str, tuple[int, int, int]],
    fps: int = 15,
) -> str | None:
    """将追踪结果渲染为带 mask overlay 的 mp4 视频。"""
    frame_count = get_frame_count(episode_path, camera)
    if frame_count == 0:
        return None

    tmp_dir = tempfile.mkdtemp(prefix="result_vid_")

    for fidx in range(frame_count):
        img = load_frame(episode_path, camera, fidx)
        if img is None:
            continue
        for mt, masks_dict in tracked_masks.items():
            if fidx in masks_dict:
                img = overlay_mask(img, masks_dict[fidx], color=mask_colors.get(mt, (255, 0, 0)))
        Image.fromarray(img).save(Path(tmp_dir) / f"{fidx:05d}.png")

    out_path = tempfile.mktemp(suffix=".mp4", prefix="result_")
    pattern = str(Path(tmp_dir) / "%05d.png")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        "-preset", "fast",
        "-loglevel", "error",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # 清理临时帧
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    if result.returncode != 0:
        return None
    return out_path


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.35,
    border_width: int = 2,
) -> np.ndarray:
    """将 mask 以半透明彩色叠加到原图上，并绘制描边轮廓。"""
    from scipy import ndimage

    out = image.copy()
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return out

    # 半透明填充
    overlay = np.zeros_like(image)
    overlay[mask_bool] = color
    out[mask_bool] = (
        (1 - alpha) * out[mask_bool] + alpha * overlay[mask_bool]
    ).astype(np.uint8)

    # 描边: 膨胀 mask 与原 mask 取差集 = 轮廓
    dilated = ndimage.binary_dilation(mask_bool, iterations=border_width)
    border = dilated & ~mask_bool
    out[border] = color

    return out


# ── Heatmap 合成与可视化 ─────────────────────────────────────────────────────

HEATMAP_COLORS = {
    "pick": np.array([0.0, 200.0, 0.0], dtype=np.float32),
    "place": np.array([220.0, 40.0, 40.0], dtype=np.float32),
    "push": np.array([0.0, 128.0, 255.0], dtype=np.float32),
    "pull": np.array([255.0, 165.0, 0.0], dtype=np.float32),
    "press": np.array([200.0, 0.0, 200.0], dtype=np.float32),
}
HEATMAP_VIS_TILE_WIDTH = 320
HEATMAP_VIS_ALPHA = 0.5


def _heatmap_dir(episode_path: str, camera: str) -> Path:
    """获取 heatmap 目录，自动适配格式。"""
    fmt = detect_format(episode_path)
    if fmt == "stereo":
        return Path(episode_path) / "heatmap" / camera / "left"
    else:
        return Path(episode_path) / "heatmap" / camera


def get_heatmap_status(episode_path: str) -> dict[str, bool]:
    """检查已保存的 heatmap 状态。返回 {camera: bool}。"""
    cameras = list_cameras(episode_path)
    status = {}
    for cam in cameras:
        hm_dir = _heatmap_dir(episode_path, cam)
        status[cam] = hm_dir.is_dir() and any(hm_dir.glob("*.npz"))
    return status


def merge_masks_to_heatmap(episode_path: str) -> str:
    """将所有相机的 mask PNG 合并为 heatmap .npz，并生成可视化。

    返回操作结果描述字符串。
    """
    ep = Path(episode_path)
    cameras = list_cameras(episode_path)
    if not cameras:
        return "未检测到相机"

    logs = []
    for cam in cameras:
        # 查找 mask 源目录
        fmt = detect_format(episode_path)
        if fmt == "stereo":
            types_parent = ep / "mask" / cam / "left"
        else:
            types_parent = ep / "mask" / cam

        if not types_parent.is_dir():
            logs.append(f"  {cam}: 无 mask 目录，跳过")
            continue

        # 收集所有帧名
        all_frames: set[str] = set()
        for mt in MASK_TYPES:
            mt_dir = types_parent / mt
            if mt_dir.is_dir():
                all_frames.update(p.stem for p in mt_dir.glob("*.png"))

        if not all_frames:
            logs.append(f"  {cam}: 无 mask 文件，跳过")
            continue

        out_dir = _heatmap_dir(episode_path, cam)
        out_dir.mkdir(parents=True, exist_ok=True)

        sorted_frames = sorted(all_frames)
        count = 0
        for frame_name in sorted_frames:
            masks = {}
            h, w = 0, 0
            for mt in MASK_TYPES:
                png_path = types_parent / mt / f"{frame_name}.png"
                if png_path.exists():
                    img = np.array(Image.open(png_path).convert("L"))
                    masks[mt] = (img > 127).astype(np.uint8)
                    h, w = img.shape
                else:
                    masks[mt] = None

            if h == 0 or w == 0:
                continue

            for mt in MASK_TYPES:
                if masks[mt] is None:
                    masks[mt] = np.zeros((h, w), dtype=np.uint8)

            np.savez(out_dir / f"{frame_name}.npz", **masks)
            count += 1

        logs.append(f"  {cam}: {count} 帧 heatmap → {out_dir}")

    # 生成可视化（所有相机的图都放在 heatmap_vis/ 下）
    vis_dir = ep / "heatmap_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    for cam in cameras:
        _build_heatmap_vis_for_camera(episode_path, cam, vis_dir)

    return "\n".join(logs) if logs else "无数据"


def _select_4_frames(n: int) -> list[int]:
    """从 n 帧中选 4 帧：首尾 + 中间均匀 2 帧。"""
    if n <= 0:
        return []
    if n <= 4:
        return list(range(n))
    mid1 = n // 3
    mid2 = 2 * n // 3
    return [0, mid1, mid2, n - 1]


def _build_heatmap_vis_for_camera(
    episode_path: str, camera: str, vis_dir: Path
):
    """为单个相机生成 4 行 2 列对比图（左原图，右叠加 heatmap）。"""
    hm_dir = _heatmap_dir(episode_path, camera)
    frames_dir = get_frames_dir(episode_path, camera)

    if not hm_dir.is_dir():
        return

    sorted_frames = sorted(p.stem for p in hm_dir.glob("*.npz"))
    if not sorted_frames:
        return

    selected_indices = _select_4_frames(len(sorted_frames))
    rows: list[tuple[Image.Image, Image.Image]] = []  # (原图, 叠加图)

    for sel_idx in selected_indices:
        frame_name = sorted_frames[sel_idx]
        rgb_path = frames_dir / f"{frame_name}.png"
        npz_path = hm_dir / f"{frame_name}.npz"
        if not rgb_path.exists() or not npz_path.exists():
            continue

        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        data = np.load(npz_path)
        label = f"f={frame_name}"
        orig_img = _annotate_image(Image.fromarray(rgb), label)
        overlay_img = _render_heatmap_overlay(rgb, data, f"heatmap f={frame_name}")
        rows.append((orig_img, overlay_img))

    if not rows:
        return

    # 拼接：4 行 x 2 列
    tile_w = HEATMAP_VIS_TILE_WIDTH
    first = rows[0][0]
    tile_h = max(1, int(round(first.height * (tile_w / float(first.width)))))

    canvas = Image.new("RGB", (tile_w * 2, tile_h * len(rows)), color=(24, 24, 24))
    for row_idx, (orig, overlay) in enumerate(rows):
        canvas.paste(orig.resize((tile_w, tile_h), resample=Image.BILINEAR), (0, row_idx * tile_h))
        canvas.paste(overlay.resize((tile_w, tile_h), resample=Image.BILINEAR), (tile_w, row_idx * tile_h))

    canvas.save(vis_dir / f"{camera}.png")


def _annotate_image(image: Image.Image, label: str) -> Image.Image:
    """在图片左上角添加标注文字。"""
    draw = ImageDraw.Draw(image)
    label_width = max(120, 8 * len(label) + 12)
    draw.rectangle((0, 0, label_width, 24), fill=(0, 0, 0))
    draw.text((6, 4), label, fill=(255, 255, 255))
    return image


def _render_heatmap_overlay(
    rgb: np.ndarray, heatmap_data, label: str
) -> Image.Image:
    """将 heatmap 各通道以半透明彩色叠加到 RGB 图上。"""
    overlay = rgb.astype(np.float32).copy()
    for mt in MASK_TYPES:
        if mt not in heatmap_data:
            continue
        mask = heatmap_data[mt] > 0
        if not np.any(mask):
            continue
        color = HEATMAP_COLORS[mt]
        overlay[mask] = (1.0 - HEATMAP_VIS_ALPHA) * overlay[mask] + HEATMAP_VIS_ALPHA * color

    image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")
    return _annotate_image(image, label)
