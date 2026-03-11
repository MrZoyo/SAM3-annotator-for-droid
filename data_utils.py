"""数据集读取、帧加载、lang.txt 读写、mask 保存工具。"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

MASK_TYPES = ["pick", "place", "push", "pull", "press"]

MASK_TYPE_CN = {
    "pick": "抓取目标",
    "place": "放置目标",
    "push": "推目标",
    "pull": "拉目标",
    "press": "按目标",
}


def list_cameras(episode_path: str) -> list[str]:
    """自动检测 episode 下的相机目录列表。"""
    stereo_dir = Path(episode_path) / "rgb_stereo_valid"
    if not stereo_dir.is_dir():
        return []
    return sorted(
        d.name
        for d in stereo_dir.iterdir()
        if d.is_dir() and (d / "left").is_dir()
    )


def list_episodes(dataset_root: str) -> list[str]:
    """列出数据集根目录下所有 episode 目录名。"""
    root = Path(dataset_root)
    if not root.is_dir():
        return []
    episodes = sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir() and (d / "rgb_stereo_valid").is_dir()
    )
    return episodes


def get_frames_dir(episode_path: str, camera: str) -> Path:
    """获取指定 episode + camera 的帧目录。"""
    return Path(episode_path) / "rgb_stereo_valid" / camera / "left"


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
    out_dir = Path(episode_path) / "mask" / camera / "left" / mask_type
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
        mask_dir = Path(episode_path) / "mask" / camera / "left" / mt
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
            mask_dir = Path(episode_path) / "mask" / cam / "left" / mt
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
