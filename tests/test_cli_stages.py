from types import SimpleNamespace

import pytest

from calibcam.__main__ import define_process_stages


@pytest.mark.parametrize(
    ("detection", "single", "multi", "expected"),
    [
        ([], False, False, (True, False, False)),
        (False, [], False, (False, True, False)),
        (False, False, True, (False, False, True)),
        (False, False, False, (True, True, True)),
    ],
)
def test_cli_process_stage_selection(detection, single, multi, expected):
    args = SimpleNamespace(
        detection=detection,
        calibration_single=single,
        calibration_multi=multi,
    )
    opts = {
        "detection": False,
        "calibration_single": False,
        "calibration_multi": False,
    }

    result = define_process_stages(args, opts)

    assert (
        result["detection"],
        result["calibration_single"],
        result["calibration_multi"],
    ) == expected
