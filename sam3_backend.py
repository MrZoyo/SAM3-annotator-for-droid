"""SAM3 VideoPredictor 推理封装。

SAM3 使用 session-based API:
- start_session(resource_path) → session_id
- model.add_prompt(..., rel_coordinates=True) → 归一化坐标 point prompting
- propagate_in_video(session_id) → generator of (frame_idx, outputs)
- reset_session(session_id)
- close_session(session_id)

坐标系说明:
  SAM3 内部将所有帧 resize 到 1008×1008。rel_coordinates=True 时，
  点坐标应为 (x/orig_w, y/orig_h) 归一化到 [0,1]，内部乘以 1008 映射。
  我们接受原始像素坐标，在 add_points 中自动归一化。
"""

import torch
import numpy as np
from sam3.model.sam3_video_predictor import Sam3VideoPredictor


class SAM3Annotator:
    def __init__(self, checkpoint_path: str | None = None):
        self.predictor = Sam3VideoPredictor(checkpoint_path=checkpoint_path)
        self.session_id: str | None = None

    def _get_state(self):
        session = self.predictor._get_session(self.session_id)
        return session["state"]

    def init_video(self, frames_dir: str):
        """初始化视频帧，创建新 session，并预填充帧缓存。"""
        if self.session_id is not None:
            self.predictor.close_session(self.session_id)

        result = self.predictor.start_session(resource_path=frames_dir)
        self.session_id = result["session_id"]

        # 预填充空的 cached_frame_outputs
        state = self._get_state()
        num_frames = state["num_frames"]
        if "cached_frame_outputs" not in state:
            state["cached_frame_outputs"] = {}
        for fidx in range(num_frames):
            if fidx not in state["cached_frame_outputs"]:
                state["cached_frame_outputs"][fidx] = {}

    def add_points(
        self,
        frame_idx: int,
        obj_id: int,
        points: list[list[float]],
        labels: list[int],
        orig_w: int,
        orig_h: int,
    ) -> np.ndarray:
        """添加正/负点（原始像素坐标），自动归一化后传给 SAM3。

        Args:
            frame_idx: 帧索引
            obj_id: 对象 ID（不同 mask 类型用不同 ID）
            points: [[x1,y1], [x2,y2], ...] 原始图像像素坐标
            labels: [1, 0, ...] 1=positive, 0=negative
            orig_w: 原始图像宽度
            orig_h: 原始图像高度

        Returns:
            binary mask (H, W) bool array
        """
        state = self._get_state()

        # 将像素坐标归一化到 [0, 1]，SAM3 内部会乘以 1008 映射到内部空间
        norm_points = [[x / orig_w, y / orig_h] for x, y in points]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            frame_idx_out, outputs = self.predictor.model.add_prompt(
                inference_state=state,
                frame_idx=frame_idx,
                points=norm_points,
                point_labels=labels,
                obj_id=obj_id,
                rel_coordinates=True,
            )

        # SAM3 的 tracker 点路径不会设置 previous_stages_out，
        # 但 propagate_in_video 依赖它判断是否有 prompt，需要手动标记
        if state["previous_stages_out"][frame_idx] is None:
            state["previous_stages_out"][frame_idx] = "_THIS_FRAME_HAS_OUTPUTS_"

        obj_ids = outputs["out_obj_ids"]
        binary_masks = outputs["out_binary_masks"]

        for i, oid in enumerate(obj_ids):
            if oid == obj_id:
                return binary_masks[i]

        # 未找到（可能被抑制），返回全零 mask
        if len(binary_masks) > 0:
            h, w = binary_masks.shape[1], binary_masks.shape[2]
        else:
            h, w = orig_h, orig_w
        return np.zeros((h, w), dtype=bool)

    def track_all(self) -> dict[int, dict[int, np.ndarray]]:
        """传播追踪所有对象到所有帧。

        Returns:
            {frame_idx: {obj_id: binary_mask (H, W)}}
        """
        results: dict[int, dict[int, np.ndarray]] = {}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for item in self.predictor.propagate_in_video(
                session_id=self.session_id,
                propagation_direction="both",
                start_frame_idx=None,
                max_frame_num_to_track=None,
            ):
                frame_idx = item["frame_index"]
                outputs = item["outputs"]
                if outputs is None:
                    continue
                obj_ids = outputs["out_obj_ids"]
                binary_masks = outputs["out_binary_masks"]

                frame_masks = {}
                for i, oid in enumerate(obj_ids):
                    frame_masks[int(oid)] = binary_masks[i]
                if frame_masks:
                    results[frame_idx] = frame_masks

        return results

    def reset(self):
        """重置当前 session 状态。"""
        if self.session_id is not None:
            self.predictor.reset_session(self.session_id)

    def close(self):
        """关闭当前 session。"""
        if self.session_id is not None:
            self.predictor.close_session(self.session_id)
            self.session_id = None
