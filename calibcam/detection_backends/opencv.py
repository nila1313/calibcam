from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import cv2
import numpy as np

from calibcam.calibrator_opts import finalize_aruco_detector_opts

from .base import DetectionBackend

if TYPE_CHECKING:
    from calibcamlib import Board


class OpenCVBackend(DetectionBackend):
    """OpenCV ChArUco detection (default backend).

    The frame passed to detect_frame is expected to be preprocessed already
    (see detection.preprocess_frame).
    """

    def create_context(
        self,
        video: str | Path,
        board: Board,
        opts: Mapping[str, Any],
    ) -> Any:
        # Build the detector configuration once per video instead of for every frame
        board_params = board.get_board_params()
        detection_opts = opts["detection_opts"]

        dictionary = cv2.aruco.getPredefinedDictionary(board_params["dictionary_type"])
        parameters = finalize_aruco_detector_opts(detection_opts["aruco_detect"])

        return {
            "detector": cv2.aruco.ArucoDetector(dictionary, parameters["parameters"]),
            "board_obj": board.get_cv2_board(),
            "refine_opts": finalize_aruco_detector_opts(detection_opts["aruco_refine"]),
            "interpolate_opts": detection_opts["aruco_interpolate"],
        }

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        context: Any,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        board_obj = context["board_obj"]

        # corner detection
        corners, ids, rejected_img_points = context["detector"].detectMarkers(frame)
        # corners, ids, rejected_img_points = \
        #     cv2.aruco.detectMarkers(frame,  # noqa
        #                             cv2.aruco.getPredefinedDictionary(board_params['dictionary_type']),  # noqa
        #                             )

        if len(corners) == 0:
            return None, None

        # corner refinement
        corners_ref, ids_ref = \
            cv2.aruco.refineDetectedMarkers(frame,  # noqa
                                            board_obj,
                                            corners,
                                            ids,
                                            rejected_img_points,
                                            **context["refine_opts"])[0:2]

        # corner interpolation
        _, charuco_corners, charuco_ids = \
            cv2.aruco.interpolateCornersCharuco(corners_ref,  # noqa
                                                ids_ref,
                                                frame,
                                                board_obj,
                                                **context["interpolate_opts"])

        if charuco_corners is None:
            return None, None

        return charuco_corners, charuco_ids
