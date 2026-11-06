import importlib
from importlib.metadata import entry_points

from .base import DetectionBackend

_BUILTIN = {"opencv": "calibcam.detection_backends.opencv:OpenCVBackend"}
GROUP = "calibcam.detection_backends"


def available() -> list[str]:
    return sorted(set(_BUILTIN) | {ep.name for ep in entry_points(group=GROUP)})


def get_detection_backend(name: str) -> type[DetectionBackend]:
    if name in _BUILTIN:
        module, cls = _BUILTIN[name].split(":")
        return getattr(importlib.import_module(module), cls)

    for ep in entry_points(group=GROUP):
        if ep.name == name:
            return ep.load()

    raise ValueError(f"Unknown detection backend {name!r}; available: {available()}")
