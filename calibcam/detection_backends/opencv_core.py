import cv2

from calibcam.calibrator_opts import finalize_aruco_detector_opts


def create_context(video, board, opts):
    """Create per-video state for OpenCV ChArUco detection."""
    return {
        "board": board,
        "board_params": board.get_board_params(),
        "opts": opts,
    }


def detect_frame(frame, frame_idx, context):
    """Detect ChArUco corners in one frame with OpenCV.

    The frame is expected to be preprocessed already (see detection.preprocess_frame).
    """
    board = context["board"]
    board_params = context["board_params"]
    opts = context["opts"]

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
