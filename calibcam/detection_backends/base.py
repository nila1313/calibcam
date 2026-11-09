from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Mapping

import numpy as np
from calibcamlib import Board


class DetectionBackend(ABC):
    """Interface for ChArUco detection backends."""

    default_opts: ClassVar[dict[str, Any]] = {}

    # Whether calibcam applies its common preprocessing (gamma correction, CLAHE,
    # color conversion; see detection.preprocess_frame) before detect_frame is called.
    # Backends that expect raw frames set this to False.
    uses_preprocessing: ClassVar[bool] = True

    @abstractmethod
    def create_context(
        self,
        video: str | Path,
        board: Board,
        opts: Mapping[str, Any],
    ) -> Any:
        """Initialize the detector state for one video."""

    @abstractmethod
    def detect_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        context: Any,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return ChArUco corners and IDs for one frame."""
