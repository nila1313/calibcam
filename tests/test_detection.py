import tempfile
import unittest

import numpy as np
from scipy.io import savemat

from calibcam.calibrator_opts import get_default_opts
from calibcam.detection import detect_corners
from calibcamlib import Detections


class TestDetection(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)

    def test_default_opts_are_mat_serializable(self):
        opts = get_default_opts(1, do_fill=True)

        with tempfile.NamedTemporaryFile(suffix=".mat") as tmp:
            savemat(
                tmp.name,
                {"opts": opts},
            )


    def test_unsupported_detection_backend_raises(self):
        opts = get_default_opts(1, do_fill=True)
        opts["detection_opts"]["backend"] = "does_not_exist"

        with self.assertRaisesRegex(ValueError, "Unknown detection backend"):
            detect_corners(
                rec_file_names=["fake_video.mp4"],
                boards=[object()],
                opts=opts,
            )



    def test_detection_merge(self):
        detections_lists = [
            {
                'marker_coords': np.array([
                    [[1, 2], [3, 4]],
                    [[5, 6], [7, 8]],
                    [[9, 10], [11, 12]],
                ]),
                'marker_ids': [[10, 13, 14], [11, 13, 15], [11, 13, 14]],
                'detection_idxs': [0, 3, 4],
                'frame_idxs': [0, 3, 4],
            }, {
                'marker_coords': np.array([
                    [[13, 14], [15, 16]],
                    [[17, 18], [19, 20]],
                    [[21, 22], [23, 24]],
                ]),
                'marker_ids': [[0, 3, 14], [1, 3, 15], [1, 3, 14]],
                'detection_idxs': [1, 3, 5],
                'frame_idxs': [11, 13, 15],
            }
        ]

        detections = [Detections.from_list(d) for d in detections_lists]

        detections_all = sum(detections, Detections())

        markers_all = [0, 1, 3, 10, 11, 13, 14, 15]
        frames_all = [0, 1, 3, 4, 5]

        d_marker_coords = detections_all.to_array()["marker_coords"]

        assert len(d_marker_coords) == 2
        assert len(markers_all) == d_marker_coords.shape[2]
        assert len(frames_all) == d_marker_coords.shape[1]

        for i_d, (d, d_list) in enumerate(zip(detections, detections_lists)):
            fr_mask = np.isin(frames_all, d_list["detection_idxs"])
            m_mask = np.isin(markers_all, d_list["marker_ids"])

            assert np.all(
                (d.to_array()["marker_coords"][0] == d_marker_coords[i_d][fr_mask][:, m_mask]) |
                np.isnan(d.to_array()["marker_coords"][0]) |
                np.isnan(d_marker_coords[i_d][fr_mask][:, m_mask])
            )

    def test_detection_sizes(self):
        detections_lists = [
            {
                'marker_coords': np.array([
                    [[1, 2], [3, 4]],
                    [[5, 6], [7, 8]],
                    [[9, 10], [11, 12]],
                ]),
                'marker_ids': [[10, 13, 14], [11, 13, 15], [11, 13, 14]],
                'detection_idxs': [0, 3, 4],
                'frame_idxs': [0, 3, 4],
            }, {
                'marker_coords': np.array([
                    [[13, 14], [15, 16]],
                    [[17, 18], [19, 20]],
                    [[21, 22], [23, 24]],
                ]),
                'marker_ids': [[0, 3, 14], [1, 3, 15], [1, 3, 14]],
                'detection_idxs': [1, 3, 5],
                'frame_idxs': [11, 13, 15],
            }
        ]

        detections = [Detections.from_list(d) for d in detections_lists]
        detections_all = sum(detections, Detections())

        assert detections_all.get_n_frames() == 5
        assert detections_all.get_n_markers() == 8
        assert np.all(detections_all.get_n_detections_frames() == np.array([[0, 0, 0, 1, 2, 3, 2, 1],
                                                                            [1, 2, 3, 0, 0, 0, 2, 1]]))
        assert np.all(detections_all.get_n_detections_markers() == np.array([[3, 0, 3, 3, 0],
                                                                             [0, 3, 3, 0, 3]]))
        assert np.all(detections_all.to_array()["frame_idxs"] == np.array([[0, -1, 3, 4, -1],
                                                                           [-1, 11, 13, -1, 15]]))
