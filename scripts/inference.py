"""Preserve source pixels and image size across fusion inference."""

import torch.nn.functional as F


def pad_image_pair(visible, infrared, multiple=16):
    """Pad only the right/bottom edges and return the original (height, width).

    Attributes should be measured on the original tensors before this step.
    Replication also supports images smaller than the padding width/height.
    """
    if not isinstance(multiple, int) or isinstance(multiple, bool) or multiple <= 0:
        raise ValueError("Padding multiple must be a positive integer.")
    if visible.ndim != 4 or visible.shape[0] == 0 or visible.shape[1] != 3:
        raise ValueError("Expected a nonempty RGB batch [B,3,H,W], got {}".format(
            tuple(visible.shape)
        ))
    if visible.shape != infrared.shape:
        raise ValueError("Original paired image shapes must match, got {} and {}".format(
            tuple(visible.shape), tuple(infrared.shape)
        ))
    height, width = visible.shape[-2:]
    if height == 0 or width == 0:
        raise ValueError("Original paired images must have positive height and width.")
    padding = (0, (-width) % multiple, 0, (-height) % multiple)
    if padding[1] or padding[3]:
        visible = F.pad(visible, padding, mode="replicate")
        infrared = F.pad(infrared, padding, mode="replicate")
    return visible, infrared, (height, width)


def crop_to_original(output, original_size):
    """Remove inference padding without resampling the model output."""
    height, width = original_size
    if (output.ndim != 4 or height <= 0 or width <= 0
            or output.shape[-2] < height or output.shape[-1] < width):
        raise ValueError("Cannot crop output shape {} to original size {}".format(
            tuple(output.shape), tuple(original_size)
        ))
    return output[..., :height, :width]
