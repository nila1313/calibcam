import unittest
from unittest.mock import patch

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

    def test_opencv_forwards_calls(self):
        backend = get_detection_backend("opencv")()
        board = object()
        opts = {}
        context = {"ready": True}

        with patch(
            "calibcam.detection_backends.opencv.opencv_core.create_context",
            return_value=context,
        ) as create:
            self.assertIs(backend.create_context("test.mp4", board, opts), context)
            create.assert_called_once_with("test.mp4", board, opts)

        frame = object()
        with patch(
            "calibcam.detection_backends.opencv.opencv_core.detect_frame",
            return_value=(None, None),
        ) as detect:
            self.assertEqual(backend.detect_frame(frame, 12, context), (None, None))
            detect.assert_called_once_with(frame, 12, context)
