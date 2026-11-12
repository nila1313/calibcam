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
        backend_class = getattr(importlib.import_module(module), cls)
    else:
        for ep in entry_points(group=GROUP, name=name):
            backend_class = ep.load()
            break
        else:
            raise ValueError(f"Unknown detection backend {name!r}; available: {available()}")

    if not (isinstance(backend_class, type) and issubclass(backend_class, DetectionBackend)):
        raise TypeError(f"Detection backend {name!r} does not implement DetectionBackend")

    return backend_class
