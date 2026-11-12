from pathlib import Path

import cv2


def existing_image_names(path):
    return {entry.name.casefold() for entry in Path(path).iterdir()}


def reserve_image_stem(stem, endings, used_names):
    """Reserve the filenames for one image and its optional reference images."""
    candidate = stem
    number = 2
    names = {(candidate + ending).casefold() for ending in endings}
    while not used_names.isdisjoint(names):
        candidate = f"{stem}__{number}"
        number += 1
        names = {(candidate + ending).casefold() for ending in endings}
    used_names.update(names)
    return candidate


def write_image(path, image):
    """Encode with OpenCV and publish without replacing an existing image."""
    path = Path(path)
    try:
        success, encoded = cv2.imencode(path.suffix, image)
    except cv2.error as exc:
        raise OSError(f"Could not encode image for {path}") from exc
    if not success:
        raise OSError(f"Could not encode image for {path}")

    try:
        output = path.open("xb")
        try:
            with output:
                output.write(encoded.tobytes())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
    except FileExistsError:
        raise
    except OSError as exc:
        raise OSError(f"Could not write image to {path}: {exc}") from exc
