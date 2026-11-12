import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from calibcamlib import Detections

from calibcam.detection_backends import (
    DetectionBackend,
    available,
    get_detection_backend,
)


class TestDetectionBackends(unittest.TestCase):
    def test_available_does_not_import_backend(self):
        with patch("calibcam.detection_backends.importlib.import_module") as importer:
            self.assertIn("opencv", available())
            importer.assert_not_called()

    def test_opencv_implements_interface(self):
        backend_class = get_detection_backend("opencv")
        self.assertTrue(issubclass(backend_class, DetectionBackend))
        backend_class()

    def test_unknown_backend(self):
        with self.assertRaisesRegex(ValueError, "Unknown detection backend"):
            get_detection_backend("unknown_test_backend")

    def test_non_backend_entry_point_is_rejected(self):
        class NotABackend:
            pass

        entry_point = MagicMock()
        entry_point.load.return_value = NotABackend
        with patch(
            "calibcam.detection_backends.entry_points", return_value=[entry_point]
        ) as find:
            with self.assertRaisesRegex(TypeError, "does not implement DetectionBackend"):
                get_detection_backend("not_a_backend")
        find.assert_called_once_with(
            group="calibcam.detection_backends", name="not_a_backend"
        )

    def test_opencv_detector_is_built_once_per_video(self):
        import cv2
        import numpy as np

        from calibcam.calibrator_opts import get_default_opts
        from calibcamlib import Board

        root = Path(__file__).resolve().parents[1]
        board = Board(np.load(root / "boards/bboboard-v4.npy", allow_pickle=True).item())
        opts = get_default_opts(1, do_fill=True)
        backend = get_detection_backend("opencv")()
        frame = cv2.imread(str(root / "tests/sample_images_1/1295.png"))

        with patch.object(
            cv2.aruco, "ArucoDetector", wraps=cv2.aruco.ArucoDetector
        ) as make_detector:
            context = backend.create_context("sample", board, opts)
            for frame_idx in range(3):
                backend.detect_frame(frame, frame_idx, context)
        make_detector.assert_called_once()

    def test_detect_corners_only_resolves_and_instantiates(self):
        from copy import deepcopy

        from calibcam import detection
        from calibcamlib import Detections

        class CustomBackend(get_detection_backend("opencv")):
            default_opts = {"work_dir": "default"}

        opts = {
            "detection_opts": {"backend": "custom", "custom": {"work_dir": "user"}},
            "frames_offsets": False,
            "frames_lists": False,
            "frames_start": 0,
            "frames_end": 1,
            "frames_step": 1,
            "parallelize": False,
        }
        original_opts = deepcopy(opts)

        with (
            patch.object(
                detection, "get_detection_backend", return_value=CustomBackend
            ) as resolve,
            patch.object(
                detection, "detect_corners_cam", return_value=Detections()
            ) as worker,
            patch.object(detection.filtergraph, "get_reader", return_value=object()),
            patch.object(detection.camfunctions, "get_n_frames_from_reader", return_value=1),
        ):
            detection.detect_corners(
                ["left.mp4", "right.mp4"], [object(), object()], opts
            )

        resolve.assert_called_once_with("custom")
        self.assertEqual(worker.call_count, 2)
        calls = worker.call_args_list
        self.assertIsInstance(calls[0].kwargs["backend"], CustomBackend)
        self.assertIs(calls[0].kwargs["backend"], calls[1].kwargs["backend"])
        for call in calls:
            self.assertIs(call.args[1], opts)  # opts are passed through unchanged
        self.assertEqual(opts, original_opts)

    def test_build_options_merges_backend_defaults_into_written_opts(self):
        import tempfile
        from importlib import import_module
        from types import SimpleNamespace

        import yaml

        cli = import_module("calibcam.__main__")

        class CustomBackend(get_detection_backend("opencv")):
            default_opts = {
                "work_dir": "default",
                "nested": {"a": 1, "b": 2},
            }

        original_defaults = {"work_dir": "default", "nested": {"a": 1, "b": 2}}
        opts = {
            "detection_opts": {
                "backend": "custom",
                "custom": {"work_dir": Path("/tmp/user"), "nested": {"b": 9}},
            },
        }

        with tempfile.TemporaryDirectory() as write_dir:
            args = SimpleNamespace(opts=[], write_opts=[write_dir])
            with (
                patch.object(cli, "build_args_into_opts", return_value=opts),
                patch.object(cli, "get_detection_backend", return_value=CustomBackend) as resolve,
            ):
                result = cli.build_options(args, n_cams=2)
            with open(Path(write_dir) / "opts.yml") as file:
                written = yaml.safe_load(file)

        resolve.assert_called_once_with("custom")
        expected = {"work_dir": "/tmp/user", "nested": {"a": 1, "b": 9}}
        self.assertEqual(result["detection_opts"]["custom"], expected)
        self.assertEqual(written["detection_opts"]["custom"], expected)
        self.assertEqual(CustomBackend.default_opts, original_defaults)

    def test_warning_when_backend_skips_enabled_preprocessing(self):
        from calibcam import detection
        from calibcam.calibrator_opts import get_default_opts

        class RawBackend(get_detection_backend("opencv")):
            uses_preprocessing = False

        opts = get_default_opts(1, do_fill=True)  # CLAHE is on by default
        opts["parallelize"] = False

        def run(expect_warning):
            with (
                patch.object(detection, "get_detection_backend", return_value=RawBackend),
                patch.object(detection, "detect_corners_cam", return_value=Detections()),
                patch.object(detection.filtergraph, "get_reader", return_value=object()),
                patch.object(detection.camfunctions, "get_n_frames_from_reader",
                             return_value=1),
            ):
                if expect_warning:
                    with self.assertLogs(detection.logger, level="WARNING") as logs:
                        detection.detect_corners(["video.mp4"], [object()], opts)
                    return logs.output
                with self.assertNoLogs(detection.logger, level="WARNING"):
                    detection.detect_corners(["video.mp4"], [object()], opts)
                return []

        output = run(expect_warning=True)
        self.assertEqual(len(output), 1)
        self.assertIn("clahe", output[0])

        opts["detection_opts"]["clahe"] = False
        run(expect_warning=False)

    def test_injected_backend_frame_is_stored(self):
        import numpy as np
        from unittest.mock import MagicMock

        from calibcam import detection
        from calibcam.calibrator_opts import get_default_opts

        board = MagicMock()
        board.get_board_params.return_value = {"boardWidth": 5}
        board.get_board_ids.return_value = [0]
        opts = get_default_opts(1, do_fill=True)
        opts["RC_reject_corners"] = False

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        reader = MagicMock()
        reader.get_data.return_value = frame
        backend = MagicMock(spec=DetectionBackend)
        backend.uses_preprocessing = False  # check that the raw frame is passed through
        context = {"ready": True}
        backend.create_context.return_value = context
        backend.detect_frame.return_value = (
            np.array([[10, 10], [20, 10], [10, 20], [20, 20], [10, 30]],
                     dtype=np.float32).reshape(-1, 1, 2),
            np.array([0, 1, 4, 5, 8], dtype=np.int32).reshape(-1, 1),
        )

        with (
            patch.object(detection.filtergraph, "get_reader", return_value=reader),
            patch.object(
                detection.camfunctions, "get_header_from_reader",
                return_value={"offset": (0, 0)},
            ),
            patch.object(
                detection.camfunctions, "get_n_frames_from_reader",
                return_value=1,
            ),
        ):
            detections = detection.detect_corners_cam(
                "fake_video.mp4", opts, board, [0], backend=backend
            )

        backend.create_context.assert_called_once_with("fake_video.mp4", board, opts)
        backend.detect_frame.assert_called_once_with(frame, 0, context)
        self.assertEqual(detections.get_n_frames(), 1)
        self.assertEqual(detections.get_n_markers(), 5)
        reader.close.assert_called_once()


class TestPreprocessing(unittest.TestCase):
    """Preprocessing runs in detection.py before the backend, not inside it."""

    def _frames_received(self, uses_preprocessing):
        """Run detect_corners_cam on one fake color frame; return what the backend got."""
        import numpy as np
        from unittest.mock import MagicMock

        from calibcam import detection
        from calibcam.calibrator_opts import get_default_opts

        received = []

        class SpyBackend(get_detection_backend("opencv")):
            def create_context(self, video, board, opts):
                return {}

            def detect_frame(self, frame, frame_idx, context):
                received.append(frame)
                return (
                    np.array([[10, 10], [20, 10], [10, 20], [20, 20], [10, 30]],
                             dtype=np.float32).reshape(-1, 1, 2),
                    np.array([0, 1, 4, 5, 8], dtype=np.int32).reshape(-1, 1),
                )

        SpyBackend.uses_preprocessing = uses_preprocessing

        board = MagicMock()
        board.get_board_params.return_value = {"boardWidth": 5}
        board.get_board_ids.return_value = [0]
        opts = get_default_opts(1, do_fill=True)  # CLAHE is on by default
        opts["RC_reject_corners"] = False

        raw = np.random.default_rng(0).integers(0, 255, (100, 100, 3), dtype=np.uint8)
        reader = MagicMock()
        reader.get_data.return_value = raw.copy()

        with (
            patch.object(detection.filtergraph, "get_reader", return_value=reader),
            patch.object(detection.camfunctions, "get_header_from_reader",
                         return_value={"offset": (0, 0)}),
            patch.object(detection.camfunctions, "get_n_frames_from_reader",
                         return_value=1),
        ):
            detection.detect_corners_cam("fake_video.mp4", opts, board, [0],
                                         backend=SpyBackend())
        return raw, opts, received

    def test_backend_receives_preprocessed_frame(self):
        import numpy as np

        from calibcam.detection import preprocess_frame

        raw, opts, received = self._frames_received(True)
        self.assertEqual(len(received), 1)
        np.testing.assert_array_equal(received[0], preprocess_frame(raw.copy(), opts))

    def test_backend_can_opt_out_of_preprocessing(self):
        import numpy as np

        raw, opts, received = self._frames_received(False)
        self.assertEqual(len(received), 1)
        np.testing.assert_array_equal(received[0], raw)

    def test_opencv_results_match_reference(self):
        """OpenCV detections on sample images must not change (reference recorded
        with the pre-refactor code, preprocessing inside the OpenCV backend)."""
        from pathlib import Path

        import cv2
        import numpy as np

        from calibcam.calibrator_opts import get_default_opts
        from calibcam.detection import preprocess_frame
        from calibcamlib import Board

        root = Path(__file__).resolve().parents[1]
        reference = np.load(root / "tests/opencv_backend_reference.npz")
        board = Board(np.load(root / "boards/bboboard-v4.npy", allow_pickle=True).item())
        opts = get_default_opts(1, do_fill=True)
        backend = get_detection_backend("opencv")()
        context = backend.create_context("sample", board, opts)

        images = sorted(root.glob("tests/sample_images_*/*.png"))
        self.assertTrue(images, "OpenCV sample images are missing")
        for image in images:
            key = f"{image.parent.name}__{image.stem}"
            frame = preprocess_frame(cv2.imread(str(image)), opts)
            corners, ids = backend.detect_frame(frame, 0, context)
            if f"{key}__ids" in reference:
                np.testing.assert_array_equal(ids, reference[f"{key}__ids"])
                np.testing.assert_allclose(
                    corners, reference[f"{key}__corners"], atol=0.05, rtol=0
                )
            else:
                self.assertIsNone(ids)
                self.assertIsNone(corners)
