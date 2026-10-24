from . import nanodeepcharuco, opencv

_METHODS = {
    "opencv": opencv,
    "nanodeepcharuco": nanodeepcharuco,
}


def get_detection_method(name):
    """Return the configured detection method module."""
    try:
        return _METHODS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported detection backend: {name!r}"
        ) from exc
