from pathlib import Path
from typing import Any, Mapping

import numpy as np
from calibcamlib import Board

from . import opencv_core

from .base import DetectionBackend


class OpenCVBackend(DetectionBackend):
    """Adapter for the existing OpenCV ChArUco detector."""

    def create_context(
        self,
        video: str | Path,
        board: Board,
        opts: Mapping[str, Any],
    ) -> Any:
        return opencv_core.create_context(video, board, opts)

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        context: Any,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        return opencv_core.detect_frame(frame, frame_idx, context)
