import unittest
from copy import deepcopy
from io import BytesIO
from pathlib import Path

import yaml
from scipy.io import savemat

from calibcam.calibrator_opts import normalize_option_paths


class TestOptionPaths(unittest.TestCase):
    def test_paths_are_serializable_without_mutation(self):
        opts = {
            "detection_opts": {
                "custom_backend": {
                    "project_root": Path("/tmp/custom_backend"),
                    "work_dir": Path("/tmp/custom_backend/work"),
                },
            },
        }
        original = deepcopy(opts)
        normalized = normalize_option_paths(opts)

        self.assertEqual(opts, original)
        self.assertIsNot(normalized, opts)
        self.assertEqual(
            normalized["detection_opts"]["custom_backend"]["project_root"],
            "/tmp/custom_backend",
        )

        yaml.safe_dump(normalized)
        savemat(BytesIO(), {"opts": normalized})

    def test_camcalibrator_load_opts_normalizes_paths(self):
        from calibcam.camcalibrator import CamCalibrator

        opts = {
            "detection_opts": {
                "custom_backend": {"project_root": Path("/tmp/custom_backend")},
            },
        }
        original = deepcopy(opts)

        loaded = CamCalibrator.load_opts(opts, data_path=None)

        self.assertEqual(opts, original)
        self.assertEqual(
            loaded["detection_opts"]["custom_backend"]["project_root"],
            "/tmp/custom_backend",
        )
        savemat(BytesIO(), {"opts": loaded})

    def test_cli_build_options_normalizes_paths(self):
        from importlib import import_module
        from types import SimpleNamespace
        from unittest.mock import patch

        cli = import_module("calibcam.__main__")
        opts = {
            "detection_opts": {
                "custom_backend": {"project_root": Path("/tmp/custom_backend")},
            },
        }
        original = deepcopy(opts)
        args = SimpleNamespace(opts=[], write_opts=[False])

        with patch.object(cli, "build_args_into_opts", return_value=opts) as build:
            result = cli.build_options(args, n_cams=2)

        build.assert_called_once()
        self.assertEqual(opts, original)
        self.assertEqual(
            result["detection_opts"]["custom_backend"]["project_root"],
            "/tmp/custom_backend",
        )
        yaml.safe_dump(result)
