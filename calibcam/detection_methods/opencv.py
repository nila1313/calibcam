import cv2
import numpy as np

from calibcam.calibrator_opts import finalize_aruco_detector_opts


def create_context(video, board, opts):
    """Create per-video state for OpenCV ChArUco detection."""
    return {
        "board": board,
        "board_params": board.get_board_params(),
        "opts": opts,
    }


def detect_frame(frame, frame_idx, context):
    """Detect ChArUco corners in one frame with OpenCV."""
    board = context["board"]
    board_params = context["board_params"]
    opts = context["opts"]

    if opts.get("gamma_correction", None) is not None:  # TODO: Generalize this
        frame -= np.min(frame)
        frame = frame.astype(np.float64)
        frame /= np.max(frame)
        frame = np.sqrt(frame)
        frame = (frame * 255).astype(np.uint8)

    if opts["detection_opts"]["clahe"]:
        if frame.dtype != np.uint8:
            frame8 = cv2.convertScaleAbs(
                frame,
                alpha=255.0 / frame.max(),
            )
        else:
            frame8 = frame

        if frame8.ndim > 2:
            lab = cv2.cvtColor(
                frame8,
                cv2.COLOR_BGR2LAB,
            )
            l_channel = cv2.split(lab)[0]
        else:
            l_channel = frame8

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )
        frame = clahe.apply(l_channel)

    # color management
    if (
        not isinstance(opts["color_convert"], bool)
        and len(frame.shape) > 2
    ):
        frame = cv2.cvtColor(
            frame,
            opts["color_convert"],
        )

    dictionary = cv2.aruco.getPredefinedDictionary(
        board_params["dictionary_type"]
    )
    parameters = finalize_aruco_detector_opts(
        opts["detection_opts"]["aruco_detect"]
    )

    detector = cv2.aruco.ArucoDetector(
        dictionary,
        parameters["parameters"],
    )

    # corner detection
    corners, ids, rejected_img_points = detector.detectMarkers(
        frame
    )

    if len(corners) == 0:
        return None, None

    board_obj = board.get_cv2_board()

    # corner refinement
    corners_ref, ids_ref = cv2.aruco.refineDetectedMarkers(
        frame,
        board_obj,
        corners,
        ids,
        rejected_img_points,
        **finalize_aruco_detector_opts(
            opts["detection_opts"]["aruco_refine"]
        ),
    )[0:2]

    # corner interpolation
    _, charuco_corners, charuco_ids = (
        cv2.aruco.interpolateCornersCharuco(
            corners_ref,
            ids_ref,
            frame,
            board_obj,
            **opts["detection_opts"]["aruco_interpolate"],
        )
    )

    if charuco_corners is None:
        return None, None

    return charuco_corners, charuco_ids
