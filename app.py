"""DROID SAM3 视频分割标注工具 - Gradio 主应用。"""

import os
import threading

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data_utils import (
    MASK_TYPES,
    MASK_TYPE_CN,
    build_preview_video,
    build_result_video,
    delete_saved_masks,
    get_frame_count,
    get_frames_dir,
    get_heatmap_status,
    get_saved_status,
    is_episode_valid,
    list_cameras,
    list_episodes,
    load_frame,
    merge_masks_to_heatmap,
    overlay_mask,
    read_lang,
    save_lang,
    save_masks,
    set_episode_valid,
)
from sam3_backend import SAM3Annotator

# ── 全局状态 ──────────────────────────────────────────────────────────────────

DEFAULT_DATASET_ROOT = os.environ.get(
    "DROID_DATASET_ROOT",
    "/data1/zoyo/projects/droid_test1/test1",
)
MAX_CAMERAS = 6  # 预留的最大相机槽位数

sam3: SAM3Annotator | None = None
SAM3_CHECKPOINT_PATH = os.environ.get("SAM3_CHECKPOINT_PATH", "").strip() or None

# 每种 mask 类型的颜色（RGB）
MASK_COLORS = {
    "pick": (0, 200, 0),       # 绿色
    "place": (220, 40, 40),    # 红色
    "push": (0, 128, 255),     # 蓝色
    "pull": (255, 165, 0),     # 橙色
    "press": (200, 0, 200),    # 紫色
}
MASK_TYPE_OBJ_ID = {mt: i + 1 for i, mt in enumerate(MASK_TYPES)}

# mask 类型选项：英文 + 中文
MASK_TYPE_LABELS = {mt: f"{mt} ({MASK_TYPE_CN[mt]})" for mt in MASK_TYPES}
LABEL_TO_TYPE = {v: k for k, v in MASK_TYPE_LABELS.items()}

# 标注状态：支持多类型同时标注，点按帧存储
# points_per_type: {mask_type: {frame_idx: [(x, y, label), ...]}}
current_state = {
    "dataset_root": DEFAULT_DATASET_ROOT,
    "episode_path": "",
    "cameras": [],
    "camera": "",
    "active_mask_type": MASK_TYPES[0],
    "point_type": 1,  # 1=positive, 0=negative
    "points_per_type": {mt: {} for mt in MASK_TYPES},
    "preview_masks": {},       # {mask_type: mask} 当前帧的预览 mask
    "tracked_masks": {},       # {mask_type: {frame_idx: mask}}
    "session_initialized": False,
    "current_frame_idx": 0,
}


def _get_ep_path(episode_name: str) -> str:
    return f"{current_state['dataset_root']}/{episode_name}"


def _get_frame_points(mask_type: str, frame_idx: int) -> list:
    """获取指定类型在指定帧上的点列表。"""
    return current_state["points_per_type"][mask_type].get(frame_idx, [])


def _has_any_points() -> bool:
    """检查是否有任何帧上的标注点。"""
    return any(
        bool(frames_dict)
        for frames_dict in current_state["points_per_type"].values()
    )


def _has_points_for_type(mask_type: str) -> bool:
    """检查指定类型是否有任何帧上的标注点。"""
    return bool(current_state["points_per_type"][mask_type])


def get_sam3() -> SAM3Annotator:
    global sam3
    if sam3 is None:
        gr.Info("正在加载 SAM3 模型，首次启动需要一些时间...")
        sam3 = SAM3Annotator(checkpoint_path=SAM3_CHECKPOINT_PATH)
        gr.Info("SAM3 模型加载完成！")
    return sam3


# ── 点可视化 ──────────────────────────────────────────────────────────────────


def draw_points_on_image(image: np.ndarray, frame_idx: int, alpha: float = 0.7) -> np.ndarray:
    """在图像上绘制当前帧上所有 mask 类型的标注点（半透明）。"""
    pil_img = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except (OSError, IOError):
        label_font = ImageFont.load_default()

    radius = 9
    sym_size = 5
    a = int(255 * alpha)

    for mask_type in MASK_TYPES:
        pts = _get_frame_points(mask_type, frame_idx)
        if not pts:
            continue
        r, g, b = MASK_COLORS[mask_type]

        for pt_idx, (x, y, label) in enumerate(pts):
            x, y = int(x), int(y)
            # 白色描边圆
            draw.ellipse(
                [(x - radius - 2, y - radius - 2), (x + radius + 2, y + radius + 2)],
                fill=(255, 255, 255, a),
            )
            # 彩色填充圆
            draw.ellipse(
                [(x - radius, y - radius), (x + radius, y + radius)],
                fill=(r, g, b, a),
            )
            # 内部 +/- 符号（白色）
            draw.line(
                [(x - sym_size, y), (x + sym_size, y)],
                fill=(255, 255, 255, a), width=2,
            )
            if label == 1:
                draw.line(
                    [(x, y - sym_size), (x, y + sym_size)],
                    fill=(255, 255, 255, a), width=2,
                )

            # 旁标类型名+编号（如 pick1, place2）
            draw.text(
                (x + radius + 3, y - 8), f"{mask_type}{pt_idx + 1}",
                fill=(r, g, b, a), font=label_font,
                stroke_width=1, stroke_fill=(255, 255, 255, a),
            )

    result = Image.alpha_composite(pil_img, overlay)
    return np.array(result.convert("RGB"))


def render_frame_with_overlays(episode_path: str, camera: str, frame_idx: int) -> np.ndarray | None:
    """加载帧并叠加所有已有的 mask 和标注点。"""
    img = load_frame(episode_path, camera, frame_idx)
    if img is None:
        return None

    # 收集哪些类型在当前帧有预览 mask（预览优先于追踪）
    preview_types_on_frame = set()
    for mt, (preview_fidx, mask) in current_state["preview_masks"].items():
        if frame_idx == preview_fidx:
            preview_types_on_frame.add(mt)

    # 叠加已追踪的 masks（跳过有预览的类型）
    for mt, masks_dict in current_state["tracked_masks"].items():
        if frame_idx in masks_dict and mt not in preview_types_on_frame:
            img = overlay_mask(img, masks_dict[frame_idx], color=MASK_COLORS[mt])

    # 叠加预览 masks（覆盖追踪结果）
    for mt, (preview_fidx, mask) in current_state["preview_masks"].items():
        if frame_idx == preview_fidx:
            img = overlay_mask(img, mask, color=MASK_COLORS[mt])

    # 绘制当前帧的标注点
    img = draw_points_on_image(img, frame_idx)
    return img


def format_points_info() -> str:
    """格式化所有类型的标注点信息，按帧分组。"""
    lines = []
    for mt in MASK_TYPES:
        frames_dict = current_state["points_per_type"][mt]
        if not frames_dict:
            continue
        total_pts = sum(len(pts) for pts in frames_dict.values())
        lines.append(f"【{mt} {MASK_TYPE_CN[mt]}】({total_pts} 个点, {len(frames_dict)} 帧):")
        for fidx in sorted(frames_dict.keys()):
            pts = frames_dict[fidx]
            lines.append(f"  帧 {fidx}:")
            for i, (x, y, label) in enumerate(pts):
                ptype = "+" if label == 1 else "-"
                lines.append(f"    {i + 1}. ({x:.0f}, {y:.0f}) {ptype}")

    if not lines:
        return "暂无标注点。点击图像添加标注点。\n提示: 切换 Mask 类型后，点击添加该类型的点。"
    return "\n".join(lines)


def format_export_status(episode_name: str) -> str:
    """格式化导出状态信息。"""
    if not episode_name:
        return "请先选择数据集"
    ep_path = _get_ep_path(episode_name)
    cameras = list_cameras(ep_path)
    if not cameras:
        return f"Episode: {episode_name}\n未检测到相机"
    status = get_saved_status(ep_path)
    hm_status = get_heatmap_status(ep_path)

    valid = is_episode_valid(ep_path)
    valid_str = "✅ 可用" if valid else "❌ 不可用"

    # 表头
    hm_col = "heatmap"
    col_w = 8
    cam_w = max(len(c) for c in cameras) + 2
    header = f"{'视角':<{cam_w}}" + "".join(f"{mt:^{col_w}}" for mt in MASK_TYPES) + f"{hm_col:^{col_w + 2}}"
    sep = "-" * len(header)
    lines = [f"Episode: {episode_name}  [{valid_str}]", "", header, sep]
    for cam in cameras:
        row = f"{cam:<{cam_w}}"
        for mt in MASK_TYPES:
            check = "  ✅  " if status.get(cam, {}).get(mt) else "  ❌  "
            row += f"{check:^{col_w}}"
        hm_check = "  ✅  " if hm_status.get(cam) else "  ❌  "
        row += f"{hm_check:^{col_w + 2}}"
        lines.append(row)
    return "\n".join(lines)


# ── 顶部回调 ─────────────────────────────────────────────────────────────────


def on_load_dataset(dataset_path):
    """加载数据集路径，刷新 episode 列表。"""
    dataset_path = dataset_path.strip()
    if not dataset_path:
        gr.Warning("请输入数据集路径")
        return gr.update(), ""
    current_state["dataset_root"] = dataset_path
    episodes = list_episodes(dataset_path)
    if not episodes:
        gr.Warning(f"未找到任何 episode: {dataset_path}")
        return gr.update(choices=[], value=None), "未找到 episode"
    gr.Info(f"已加载 {len(episodes)} 个 episode")
    return gr.update(choices=episodes, value=episodes[0]), format_export_status(episodes[0])


def on_episode_select(episode_name):
    """选择 episode: 显示首帧图 + 空视频槽 + 生成按钮 + lang + camera_radio + 导出状态。"""
    n = MAX_CAMERAS
    hide = gr.update(value=None, visible=False)

    if not episode_name:
        return (
            *[hide] * n,   # images
            *[hide] * n,   # videos
            *[hide] * n,   # buttons
            "",            # lang
            "可用",         # validity_radio
            gr.update(choices=[], value=None),  # camera_radio
            None,          # annotate_img
            gr.update(maximum=0, value=0),  # annotate_slider
            format_points_info(),  # points_info
            "请先选择数据集",  # export_status
        )

    ep_path = _get_ep_path(episode_name)
    current_state["episode_path"] = ep_path
    cameras = list_cameras(ep_path)
    current_state["cameras"] = cameras
    current_state["camera"] = cameras[0] if cameras else ""
    _reset_annotation_state(reset_session=True)
    lang = read_lang(ep_path)

    img_out, vid_out, btn_out = [], [], []
    for i in range(n):
        if i < len(cameras):
            frame = load_frame(ep_path, cameras[i], 0)
            img_out.append(gr.update(value=frame, visible=True, label=cameras[i]))
            vid_out.append(gr.update(value=None, visible=True, label=cameras[i]))
            btn_out.append(gr.update(
                visible=True, value=f"生成 {cameras[i]} 视频", interactive=True
            ))
        else:
            img_out.append(hide)
            vid_out.append(hide)
            btn_out.append(hide)

    valid = is_episode_valid(ep_path)
    validity_val = "可用" if valid else "不可用"

    # Tab 2 的帧 slider 和标注图也一并重置，后台预加载 SAM3 session
    first_cam = cameras[0] if cameras else ""
    if first_cam:
        _ensure_session_async(episode_name, first_cam)
        frame_count = get_frame_count(ep_path, first_cam)
        max_idx = max(0, frame_count - 1)
        first_frame = load_frame(ep_path, first_cam, 0)
    else:
        max_idx = 0
        first_frame = None

    return (
        *img_out, *vid_out, *btn_out,
        lang,
        validity_val,
        gr.update(choices=cameras, value=first_cam or None),
        first_frame,
        gr.update(maximum=max_idx, value=0),
        format_points_info(),
        format_export_status(episode_name),
    )


def on_prev_episode(episode_name):
    """切换到上一个 episode，并自动切到 Tab 1。"""
    episodes = list_episodes(current_state["dataset_root"])
    if not episodes:
        return gr.update(), gr.update()
    if not episode_name or episode_name not in episodes:
        return gr.update(value=episodes[0]), gr.update(selected="tab1")
    idx = episodes.index(episode_name)
    prev_idx = (idx - 1) % len(episodes)
    return gr.update(value=episodes[prev_idx]), gr.update(selected="tab1")


def on_next_episode(episode_name):
    """切换到下一个 episode，并自动切到 Tab 1。"""
    episodes = list_episodes(current_state["dataset_root"])
    if not episodes:
        return gr.update(), gr.update()
    if not episode_name or episode_name not in episodes:
        return gr.update(value=episodes[0]), gr.update(selected="tab1")
    idx = episodes.index(episode_name)
    next_idx = (idx + 1) % len(episodes)
    return gr.update(value=episodes[next_idx]), gr.update(selected="tab1")


# ── Tab 1 回调 ────────────────────────────────────────────────────────────────


def on_validity_change(episode_name, validity):
    """切换 episode 可用/不可用状态。"""
    if not episode_name:
        return format_export_status(episode_name)
    ep_path = _get_ep_path(episode_name)
    valid = validity == "可用"
    old_valid = is_episode_valid(ep_path)
    if valid == old_valid:
        return format_export_status(episode_name)
    set_episode_valid(ep_path, valid)
    status = "可用" if valid else "不可用"
    gr.Info(f"已标记为 {status}")
    return format_export_status(episode_name)


def on_save_lang_and_next(episode_name, lang_text):
    if episode_name:
        save_lang(_get_ep_path(episode_name), lang_text)
        gr.Info("lang.txt 已保存！")
    return gr.update(selected="tab2")


# ── Tab 2 回调 ────────────────────────────────────────────────────────────────


def _reset_annotation_state(reset_session: bool = True):
    """重置标注状态。"""
    global _session_init_thread
    current_state["points_per_type"] = {mt: {} for mt in MASK_TYPES}
    current_state["preview_masks"] = {}
    current_state["tracked_masks"] = {}
    current_state["current_frame_idx"] = 0
    if reset_session:
        current_state["session_initialized"] = False


def on_camera_change(camera, episode_name, mask_type):
    if not episode_name or not camera:
        return None, gr.update(maximum=0, value=0), "请先选择数据集"
    ep_path = _get_ep_path(episode_name)

    # 如果是 on_episode_select 触发的（相机和 episode 都已设好，session 已在加载），跳过重复初始化
    already_set = (current_state["camera"] == camera and current_state["episode_path"] == ep_path)
    current_state["camera"] = camera

    if not already_set:
        _reset_annotation_state()
        _ensure_session_async(episode_name, camera)

    frame_count = get_frame_count(ep_path, camera)
    max_idx = max(0, frame_count - 1)
    img = load_frame(ep_path, camera, 0)
    return img, gr.update(maximum=max_idx, value=0), format_points_info()


def _parse_mask_type(mask_type_label: str) -> str:
    """从 'pick (抓取目标)' 格式中提取类型名。"""
    return LABEL_TO_TYPE.get(mask_type_label, MASK_TYPES[0])


def on_mask_type_change(mask_type):
    current_state["active_mask_type"] = _parse_mask_type(mask_type)
    return format_points_info()


def on_point_type_change(point_type):
    current_state["point_type"] = 1 if point_type == "Positive +" else 0


def on_annotate_frame_change(frame_idx, episode_name):
    if not episode_name:
        return None
    ep_path = _get_ep_path(episode_name)
    fidx = int(frame_idx)
    current_state["current_frame_idx"] = fidx
    return render_frame_with_overlays(ep_path, current_state["camera"], fidx)


def on_image_click(episode_name, frame_idx, evt: gr.SelectData):
    if not episode_name:
        return None, format_points_info()

    x, y = evt.index[0], evt.index[1]
    mt = current_state["active_mask_type"]
    label = current_state["point_type"]
    fidx = int(frame_idx)
    current_state["current_frame_idx"] = fidx

    # 按帧存储点
    if fidx not in current_state["points_per_type"][mt]:
        current_state["points_per_type"][mt][fidx] = []
    current_state["points_per_type"][mt][fidx].append((x, y, label))

    # 自动预览 mask
    _auto_preview(episode_name, fidx)

    ep_path = _get_ep_path(episode_name)
    img = render_frame_with_overlays(ep_path, current_state["camera"], fidx)
    return img, format_points_info()


def on_clear_points(episode_name, frame_idx):
    """清除当前 mask 类型在当前帧的点。"""
    mt = current_state["active_mask_type"]
    fidx = int(frame_idx)
    current_state["points_per_type"][mt].pop(fidx, None)
    current_state["preview_masks"].pop(mt, None)
    if not episode_name:
        return None, format_points_info()
    # 刷新显示（mask 已清除）
    img = render_frame_with_overlays(
        _get_ep_path(episode_name), current_state["camera"], fidx
    )
    return img, format_points_info()


def on_clear_all_points(episode_name, frame_idx):
    """清除所有标注状态（点、预览、追踪结果、session），并重新初始化 session。"""
    _reset_annotation_state()
    if not episode_name:
        return None, format_points_info()
    # 后台重新初始化 session
    _ensure_session_async(episode_name, current_state["camera"])
    fidx = int(frame_idx)
    img = render_frame_with_overlays(
        _get_ep_path(episode_name), current_state["camera"], fidx
    )
    return img, format_points_info()


def on_delete_last_point(episode_name, frame_idx):
    """撤销当前类型在当前帧的最后一个点。"""
    mt = current_state["active_mask_type"]
    fidx = int(frame_idx)
    pts = current_state["points_per_type"][mt].get(fidx, [])
    if pts:
        pts.pop()
        if not pts:
            del current_state["points_per_type"][mt][fidx]
    if not episode_name:
        return None, format_points_info()
    # 重新预览（点变了，mask 也要更新）
    _auto_preview(episode_name, fidx)
    img = render_frame_with_overlays(
        _get_ep_path(episode_name), current_state["camera"], fidx
    )
    return img, format_points_info()


_session_lock = threading.Lock()
_session_init_thread: threading.Thread | None = None
_session_version = 0  # 每次请求新 session 时递增，过期线程不生效
_session_key: tuple[str, str] = ("", "")  # 当前 session 对应的 (episode, camera)


def _init_session_sync(episode_name, camera, version: int):
    """同步初始化 SAM3 session（在后台线程中调用）。"""
    global _session_key
    ep_path = _get_ep_path(episode_name)
    annotator = get_sam3()
    with _session_lock:
        # 再次检查版本号，避免已被更新的请求覆盖
        if version != _session_version:
            return
        annotator.init_video(str(get_frames_dir(ep_path, camera)))
        current_state["session_initialized"] = True
        _session_key = (episode_name, camera)


def _ensure_session_async(episode_name, camera):
    """异步初始化 SAM3 session，立即返回不阻塞。"""
    global _session_init_thread, _session_version
    if current_state["session_initialized"] and _session_key == (episode_name, camera):
        return
    # 递增版本号，使任何正在运行的旧线程失效
    _session_version += 1
    current_state["session_initialized"] = False
    version = _session_version
    _session_init_thread = threading.Thread(
        target=_init_session_sync,
        args=(episode_name, camera, version),
        daemon=True,
    )
    _session_init_thread.start()


def _ensure_session(episode_name, camera):
    """确保 SAM3 session 已初始化且匹配当前 episode+camera（同步等待）。"""
    global _session_init_thread
    # 如果后台线程正在加载，等待完成
    if _session_init_thread is not None and _session_init_thread.is_alive():
        _session_init_thread.join()
    # 如果 session 不匹配或未初始化，重新初始化
    if not current_state["session_initialized"] or _session_key != (episode_name, camera):
        _init_session_sync(episode_name, camera, _session_version)


def _get_image_size(episode_name, camera) -> tuple[int, int]:
    """获取原始图像尺寸 (width, height)。"""
    ep_path = _get_ep_path(episode_name)
    img = load_frame(ep_path, camera, 0)
    if img is not None:
        h, w = img.shape[:2]
        return w, h
    return 640, 480


def _auto_preview(episode_name, frame_idx: int):
    """自动预览当前帧上所有有点的类型的 mask。"""
    if not episode_name:
        return

    camera = current_state["camera"]
    # 等待后台 session 加载完成（通常已经好了）
    _ensure_session(episode_name, camera)
    annotator = get_sam3()
    orig_w, orig_h = _get_image_size(episode_name, camera)
    fidx = int(frame_idx)

    for mt in MASK_TYPES:
        pts = _get_frame_points(mt, fidx)
        if pts:
            obj_id = MASK_TYPE_OBJ_ID[mt]
            points = [[p[0], p[1]] for p in pts]
            labels = [p[2] for p in pts]
            mask = annotator.add_points(fidx, obj_id, points, labels, orig_w, orig_h)
            current_state["preview_masks"][mt] = (fidx, mask)
        else:
            # 当前帧没有该类型的点了，清除预览
            if mt in current_state["preview_masks"]:
                prev_fidx, _ = current_state["preview_masks"][mt]
                if prev_fidx == fidx:
                    del current_state["preview_masks"][mt]


def on_track(episode_name, frame_idx):
    """追踪所有已标注的 mask 类型到全视频（generator 模式，更新按钮状态）。"""
    if not episode_name:
        gr.Warning("请先选择数据集")
        yield None, gr.update(maximum=0, value=0), None, gr.update()
        return

    if not _has_any_points():
        gr.Warning("请先添加标注点并预览 mask")
        yield None, gr.update(maximum=0, value=0), None, gr.update()
        return

    # 按钮变灰
    yield gr.update(), gr.update(), gr.update(), gr.update(value="正在追踪中...", interactive=False, variant="secondary")

    ep_path = _get_ep_path(episode_name)
    camera = current_state["camera"]

    # 每次追踪都重新初始化 session，确保干净的状态
    with _session_lock:
        annotator = get_sam3()
        annotator.init_video(str(get_frames_dir(ep_path, camera)))
        current_state["session_initialized"] = True
        global _session_key
        _session_key = (episode_name, camera)
    orig_w, orig_h = _get_image_size(episode_name, camera)

    # 将所有帧上的所有类型的点都 add_prompt 给 SAM3
    for mt in MASK_TYPES:
        frames_dict = current_state["points_per_type"][mt]
        if not frames_dict:
            continue
        obj_id = MASK_TYPE_OBJ_ID[mt]
        for fidx in sorted(frames_dict.keys()):
            pts = frames_dict[fidx]
            points = [[p[0], p[1]] for p in pts]
            labels = [p[2] for p in pts]
            annotator.add_points(fidx, obj_id, points, labels, orig_w, orig_h)

    all_tracked = annotator.track_all()

    # 按 mask 类型分组
    current_state["tracked_masks"] = {}
    for mt in MASK_TYPES:
        obj_id = MASK_TYPE_OBJ_ID[mt]
        mt_masks = {}
        for f_idx, obj_masks in all_tracked.items():
            if obj_id in obj_masks:
                mt_masks[f_idx] = obj_masks[obj_id]
        if mt_masks:
            current_state["tracked_masks"][mt] = mt_masks

    # 追踪完成后清除预览 mask，以追踪结果为准
    current_state["preview_masks"] = {}

    tracked_types = list(current_state["tracked_masks"].keys())
    total_frames = sum(len(m) for m in current_state["tracked_masks"].values())
    gr.Info(f"追踪完成！类型: {tracked_types}，共 {total_frames} 帧 mask。正在生成预览视频...")

    frame_count = get_frame_count(ep_path, camera)
    max_idx = max(0, frame_count - 1)
    img = render_frame_with_overlays(ep_path, camera, 0)

    # 生成结果预览视频
    vid_path = build_result_video(ep_path, camera, current_state["tracked_masks"], MASK_COLORS)

    # 按钮恢复
    yield img, gr.update(maximum=max_idx, value=0), vid_path, gr.update(value="开始追踪全部", interactive=True, variant="primary")


def on_result_slider_change(frame_idx, episode_name):
    if not episode_name:
        return None
    fidx = int(frame_idx)
    current_state["current_frame_idx"] = fidx
    return render_frame_with_overlays(
        _get_ep_path(episode_name), current_state["camera"], fidx
    )


def on_save_masks(episode_name):
    """保存所有已追踪的 mask 类型（完整序列），并返回刷新的导出状态。"""
    if not episode_name:
        gr.Warning("请先选择数据集")
        return format_export_status(episode_name)
    tracked = current_state["tracked_masks"]
    if not tracked:
        gr.Warning("请先执行追踪")
        return format_export_status(episode_name)

    ep_path = _get_ep_path(episode_name)
    camera = current_state["camera"]
    total_frames = get_frame_count(ep_path, camera)

    saved_paths = []
    for mt, masks_dict in tracked.items():
        out_dir = save_masks(ep_path, camera, mt, masks_dict, total_frames)
        saved_paths.append(f"{mt}: {out_dir} ({len(masks_dict)}/{total_frames}帧有mask)")

    gr.Info("保存完成！\n" + "\n".join(saved_paths))
    return format_export_status(episode_name)


def on_delete_saved_masks(episode_name):
    """删除当前视角所有已保存的 mask 文件。"""
    if not episode_name:
        gr.Warning("请先选择数据集")
        return format_export_status(episode_name)
    camera = current_state["camera"]
    if not camera:
        gr.Warning("请先选择视角")
        return format_export_status(episode_name)

    ep_path = _get_ep_path(episode_name)
    deleted = delete_saved_masks(ep_path, camera)
    if deleted:
        gr.Info(f"已删除 {camera} 的 mask: {', '.join(deleted)}")
    else:
        gr.Info(f"{camera} 没有已保存的 mask")
    return format_export_status(episode_name)


def on_merge_heatmap(episode_name):
    """合成 heatmap：将已保存的 mask 合并为 .npz 并生成可视化。"""
    if not episode_name:
        gr.Warning("请先选择数据集")
        return format_export_status(episode_name)
    ep_path = _get_ep_path(episode_name)
    result = merge_masks_to_heatmap(ep_path)
    gr.Info(f"Heatmap 合成完成！\n{result}")
    return format_export_status(episode_name)


def on_next_camera(episode_name, current_camera):
    cameras = current_state["cameras"]
    if not cameras or current_camera not in cameras:
        return gr.update()
    idx = cameras.index(current_camera)
    next_idx = (idx + 1) % len(cameras)
    if next_idx <= idx:
        gr.Info("所有视角已完成！")
    return cameras[next_idx]


# ── 构建 Gradio UI ──────────────────────────────────────────────────────────


def build_app():
    episodes = list_episodes(DEFAULT_DATASET_ROOT)

    # mask 类型 choices：英文 + 中文
    mask_choices = [MASK_TYPE_LABELS[mt] for mt in MASK_TYPES]

    # 颜色图例 HTML（紧凑，放在 Radio 上方）
    legend_parts = []
    for mt in MASK_TYPES:
        r, g, b = MASK_COLORS[mt]
        legend_parts.append(
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background:rgb({r},{g},{b});border-radius:50%;vertical-align:middle;'
            f'margin-right:3px;"></span>'
            f'<span style="color:rgb({r},{g},{b});font-weight:600;">{mt}</span>'
        )
    color_legend_html = " &nbsp;&nbsp; ".join(legend_parts)

    with gr.Blocks(
        title="DROID SAM3 视频分割标注工具",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("# DROID SAM3 视频分割标注工具")

        # ── 顶部固定区域：数据集路径 + Episode 选择 + 导出状态 ──
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    dataset_path_input = gr.Textbox(
                        value=DEFAULT_DATASET_ROOT,
                        label="数据集路径",
                        interactive=True,
                    )
                with gr.Row():
                    episode_dropdown = gr.Dropdown(
                        choices=episodes, value=None,
                        label="选择 Episode", interactive=True,
                    )
            with gr.Column(scale=0, min_width=100):
                gr.HTML('<div style="height:8px;"></div>')
                btn_load_dataset = gr.Button("加载路径")
                btn_prev_ep = gr.Button("← 上一个")
                btn_next_ep = gr.Button("下一个 →")
            with gr.Column(scale=2):
                export_status = gr.Textbox(
                    label="标注状态总览", lines=6, interactive=False,
                    value="请先选择数据集",
                )
                btn_refresh_status = gr.Button("刷新状态", size="sm", variant="secondary")

        with gr.Tabs() as tabs:
            # ── Tab 1: 数据预览 & 语言标注 ──
            with gr.Tab("1. 数据预览 & 语言标注", id="tab1"):
                img_slots = []
                btn_slots = []
                vid_slots = []
                with gr.Row(equal_height=True):
                    for _ in range(MAX_CAMERAS):
                        with gr.Column():
                            img_slots.append(
                                gr.Image(interactive=False, visible=False, height=240)
                            )
                            btn_slots.append(
                                gr.Button(visible=False, size="sm")
                            )
                            vid_slots.append(
                                gr.Video(interactive=False, visible=False, height=240)
                            )

                with gr.Row():
                    lang_box = gr.Textbox(
                        label="动作描述 (lang.txt)", lines=2, interactive=True,
                        scale=3,
                    )
                    validity_radio = gr.Radio(
                        choices=["可用", "不可用"], value="可用",
                        label="数据集状态", interactive=True,
                        scale=1,
                    )
                btn_next_tab = gr.Button("保存并进入标注 →", variant="primary")

            # ── Tab 2: SAM3 分割标注 ──
            with gr.Tab("2. SAM3 分割标注", id="tab2"):
                with gr.Row():
                    camera_radio = gr.Radio(
                        choices=[], value=None,
                        label="选择视角", interactive=True,
                    )
                # 颜色图例 + mask 类型选择
                gr.HTML(f'<div style="padding:4px 8px;font-size:14px;">{color_legend_html}</div>')
                with gr.Row():
                    mask_type_radio = gr.Radio(
                        choices=mask_choices, value=mask_choices[0],
                        label="标注类型", interactive=True,
                    )
                    point_type_radio = gr.Radio(
                        choices=["Positive +", "Negative -"],
                        value="Positive +", label="点类型", interactive=True,
                    )
                with gr.Row():
                    with gr.Column(scale=3):
                        annotate_slider = gr.Slider(
                            0, 0, step=1, label="帧索引", interactive=True
                        )
                        annotate_img = gr.Image(
                            label="当前帧（点击添加标注点）", interactive=False,
                        )
                    with gr.Column(scale=1):
                        points_info = gr.Textbox(
                            label="标注点列表", lines=12, interactive=False,
                            value="暂无标注点。点击图像添加标注点。\n提示: 切换 Mask 类型后，点击添加该类型的点。",
                        )
                        with gr.Row():
                            btn_del_last = gr.Button("撤销最后一个点")
                            btn_clear = gr.Button("清除当前类型")
                            btn_clear_all = gr.Button("清除全部", variant="stop")
                        btn_track = gr.Button("开始追踪全部", variant="primary")
                        gr.HTML('<div style="height:12px;"></div>')
                        btn_delete_saved = gr.Button(
                            "🗑️ 清除当前视角已保存 Mask",
                            variant="stop",
                            size="sm",
                        )

                gr.Markdown("### 追踪结果预览")
                with gr.Row(equal_height=True):
                    with gr.Column():
                        result_slider = gr.Slider(
                            0, 0, step=1, label="结果帧索引", interactive=True
                        )
                        result_img = gr.Image(label="逐帧预览", interactive=False)
                    with gr.Column():
                        result_video = gr.Video(label="追踪结果视频", interactive=False)

                with gr.Row():
                    btn_save_mask = gr.Button("保存所有 Mask", variant="primary")
                    btn_merge_heatmap = gr.Button("合成 Heatmap", variant="secondary")
                    btn_next_cam = gr.Button("下一个视角 →")

        # ── 绑定事件 ─────────────────────────────────────────────────────────

        # 顶部：加载数据集
        btn_load_dataset.click(
            on_load_dataset,
            inputs=[dataset_path_input],
            outputs=[episode_dropdown, export_status],
        )

        # 顶部：episode 选择（自动刷新导出状态 + 重置 Tab2 slider）
        episode_dropdown.change(
            on_episode_select,
            inputs=[episode_dropdown],
            outputs=[*img_slots, *vid_slots, *btn_slots, lang_box, validity_radio, camera_radio, annotate_img, annotate_slider, points_info, export_status],
        )

        # 顶部：上/下一个 episode（自动切回 Tab 1）
        btn_prev_ep.click(
            on_prev_episode,
            inputs=[episode_dropdown],
            outputs=[episode_dropdown, tabs],
        )
        btn_next_ep.click(
            on_next_episode,
            inputs=[episode_dropdown],
            outputs=[episode_dropdown, tabs],
        )

        # 顶部：手动刷新状态
        btn_refresh_status.click(
            format_export_status,
            inputs=[episode_dropdown],
            outputs=[export_status],
        )

        # Tab 1: 每个槽位的独立生成按钮
        for i in range(MAX_CAMERAS):
            def make_gen_fn(slot_idx):
                def gen_video(episode_name):
                    cameras = current_state["cameras"]
                    if not episode_name or slot_idx >= len(cameras):
                        gr.Warning("请先选择数据集")
                        return gr.update(), gr.update()

                    cam = cameras[slot_idx]
                    ep_path = _get_ep_path(episode_name)

                    yield (
                        gr.update(value=None, label=f"{cam} (转换中...)"),
                        gr.update(value=f"生成中...", interactive=False),
                    )

                    vid_path = build_preview_video(ep_path, cam)

                    yield (
                        gr.update(value=vid_path, label=cam),
                        gr.update(value=f"重新生成 {cam}", interactive=True),
                    )

                return gen_video

            btn_slots[i].click(
                make_gen_fn(i),
                inputs=[episode_dropdown],
                outputs=[vid_slots[i], btn_slots[i]],
            )

        # Tab 1: 数据集可用性切换
        validity_radio.change(
            on_validity_change,
            inputs=[episode_dropdown, validity_radio],
            outputs=[export_status],
        )

        btn_next_tab.click(
            on_save_lang_and_next,
            inputs=[episode_dropdown, lang_box],
            outputs=[tabs],
        )

        # Tab 2
        camera_radio.change(
            on_camera_change,
            inputs=[camera_radio, episode_dropdown, mask_type_radio],
            outputs=[annotate_img, annotate_slider, points_info],
        )
        mask_type_radio.change(
            on_mask_type_change, inputs=[mask_type_radio], outputs=[points_info],
        )
        point_type_radio.change(on_point_type_change, inputs=[point_type_radio])
        annotate_slider.change(
            on_annotate_frame_change,
            inputs=[annotate_slider, episode_dropdown],
            outputs=[annotate_img],
        )
        annotate_img.select(
            on_image_click,
            inputs=[episode_dropdown, annotate_slider],
            outputs=[annotate_img, points_info],
        )
        btn_del_last.click(
            on_delete_last_point,
            inputs=[episode_dropdown, annotate_slider],
            outputs=[annotate_img, points_info],
        )
        btn_clear.click(
            on_clear_points,
            inputs=[episode_dropdown, annotate_slider],
            outputs=[annotate_img, points_info],
        )
        btn_clear_all.click(
            on_clear_all_points,
            inputs=[episode_dropdown, annotate_slider],
            outputs=[annotate_img, points_info],
        )
        btn_track.click(
            on_track,
            inputs=[episode_dropdown, annotate_slider],
            outputs=[result_img, result_slider, result_video, btn_track],
        )
        result_slider.change(
            on_result_slider_change,
            inputs=[result_slider, episode_dropdown],
            outputs=[result_img],
        )
        # 保存 mask 后自动刷新导出状态
        btn_save_mask.click(
            on_save_masks,
            inputs=[episode_dropdown],
            outputs=[export_status],
        )
        btn_merge_heatmap.click(
            on_merge_heatmap,
            inputs=[episode_dropdown],
            outputs=[export_status],
        )
        btn_delete_saved.click(
            on_delete_saved_masks,
            inputs=[episode_dropdown],
            outputs=[export_status],
        )
        btn_next_cam.click(
            on_next_camera,
            inputs=[episode_dropdown, camera_radio],
            outputs=[camera_radio],
        )

    return app


if __name__ == "__main__":
    print("正在加载 SAM3 模型...")
    get_sam3()
    print("SAM3 模型加载完成！")
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
