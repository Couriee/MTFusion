"""Inference must preserve source geometry and absolute output intensity."""

import ast
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix
import torch
from torchvision.transforms import functional as F

from scripts.image_output import existing_image_names, reserve_image_stem, write_image
from scripts.inference import crop_to_original, pad_image_pair


def load_inference_functions(model):
    """Use the real entry point without importing/downloading CLIP weights."""
    path = Path(__file__).resolve().parents[1] / "test_from_dataset.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    namespace = {
        "__file__": str(path), "os": os, "Path": Path, "np": np,
        "Image": Image, "cv2": cv2, "torch": torch, "F": F,
        "graycomatrix": graycomatrix,
        "existing_image_names": existing_image_names,
        "reserve_image_stem": reserve_image_stem, "write_image": write_image,
        "pad_image_pair": pad_image_pair, "crop_to_original": crop_to_original,
        "create_model1": lambda *args, **kwargs: model,
        "clip": SimpleNamespace(load=lambda *args, **kwargs: (None, None)),
    }
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class EchoModel(torch.nn.Module):
    def forward(self, visible, infrared, attributes):
        self.visible = visible.detach().clone()
        self.infrared = infrared.detach().clone()
        self.attributes = attributes.detach().clone()
        return visible


class InferenceIOTests(unittest.TestCase):
    def test_padding_and_crop_preserve_every_source_pixel(self):
        visible = torch.arange(2 * 3 * 17 * 29).reshape(2, 3, 17, 29).float()
        infrared = visible + 1
        padded_vi, padded_ir, original_size = pad_image_pair(visible, infrared)
        self.assertEqual(padded_vi.shape, (2, 3, 32, 32))
        self.assertEqual(original_size, (17, 29))
        torch.testing.assert_close(crop_to_original(padded_vi, original_size), visible)
        torch.testing.assert_close(crop_to_original(padded_ir, original_size), infrared)
        torch.testing.assert_close(padded_vi[..., -1, -1], visible[..., -1, -1])
        torch.testing.assert_close(padded_vi[..., :17, 29:], visible[..., -1:].expand(-1, -1, -1, 3))

    def test_padding_supports_small_images_and_leaves_aligned_images_unchanged(self):
        tiny = torch.ones(1, 3, 1, 1) * 0.4
        padded, _, size = pad_image_pair(tiny, tiny)
        self.assertEqual(padded.shape, (1, 3, 16, 16))
        torch.testing.assert_close(padded, torch.full_like(padded, 0.4))
        torch.testing.assert_close(crop_to_original(padded, size), tiny)
        aligned = torch.rand(1, 3, 16, 32)
        padded, _, size = pad_image_pair(aligned, aligned)
        self.assertEqual(size, (16, 32))
        self.assertIs(padded, aligned)

    def test_mismatched_pairs_and_undersized_output_fail(self):
        visible = torch.zeros(1, 3, 17, 29)
        with self.assertRaisesRegex(ValueError, "Original paired image shapes must match"):
            pad_image_pair(visible, torch.zeros(1, 3, 18, 29))
        with self.assertRaisesRegex(ValueError, "Cannot crop"):
            crop_to_original(torch.zeros(1, 3, 16, 29), (17, 29))

    def test_export_clips_without_stretching_or_mutating_the_output(self):
        tools = load_inference_functions(EchoModel())
        image = np.array([[[-0.2, 0.4, 1.2], [0.5, 0.5, 0.5]]], dtype=np.float32)
        before = image.copy()
        with TemporaryDirectory() as directory:
            tools["save_pic"](image, directory, "brightness.png")
            decoded = cv2.imdecode(
                np.frombuffer((Path(directory) / "brightness.png").read_bytes(), dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
        np.testing.assert_array_equal(decoded[..., ::-1], [[[0, 102, 255], [128, 128, 128]]])
        np.testing.assert_array_equal(image, before)

    def test_export_preserves_a_constant_midgray_image(self):
        tools = load_inference_functions(EchoModel())
        with TemporaryDirectory() as directory:
            tools["save_pic"](np.full((2, 3, 3), 0.4, dtype=np.float32), directory, "gray.png")
            decoded = cv2.imdecode(
                np.frombuffer((Path(directory) / "gray.png").read_bytes(), dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
        np.testing.assert_array_equal(decoded, np.full((2, 3, 3), 102, dtype=np.uint8))

    def test_cpu_entry_point_uses_original_attributes_then_pads_and_crops(self):
        model = EchoModel()
        tools = load_inference_functions(model)
        pixels = (np.arange(17 * 29 * 3).reshape(17, 29, 3) % 41 + 80).astype(np.uint8)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Visible").mkdir()
            (root / "Infrared").mkdir()
            Image.fromarray(pixels).save(root / "Visible" / "scene.png")
            Image.fromarray(pixels).save(root / "Infrared" / "scene.png")
            args = SimpleNamespace(dataset_path=str(root), save_path=str(root / "output"),
                                   device="cpu", weights_path="unused.pth", curvature=None)
            with patch.object(torch, "load", return_value={"model": {}, "curvature": 1.0}):
                with redirect_stdout(io.StringIO()):
                    tools["main"](args)
            decoded = cv2.imdecode(
                np.frombuffer((root / "output" / "scene.png").read_bytes(), dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
        self.assertEqual(model.visible.device.type, "cpu")
        self.assertEqual(model.visible.shape, (1, 3, 32, 32))
        original = F.to_tensor(Image.fromarray(pixels)).unsqueeze(0)
        torch.testing.assert_close(model.attributes[:, 3], tools["calculate_contrast"](original))
        torch.testing.assert_close(model.visible[..., :17, :29], original)
        np.testing.assert_array_equal(decoded[..., ::-1], pixels)

    def test_entry_point_rejects_original_size_mismatch_before_tensor_conversion(self):
        tools = load_inference_functions(EchoModel())
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Visible").mkdir()
            (root / "Infrared").mkdir()
            Image.new("RGB", (29, 17)).save(root / "Visible" / "scene.png")
            Image.new("RGB", (29, 18)).save(root / "Infrared" / "scene.png")
            args = SimpleNamespace(dataset_path=str(root), save_path=str(root / "output"),
                                   device="cpu", weights_path="unused.pth", curvature=None)
            with patch.object(torch, "load", return_value={"model": {}}):
                with patch.object(F, "to_tensor", side_effect=AssertionError("Converted mismatched inputs")):
                    with redirect_stdout(io.StringIO()):
                        with self.assertRaisesRegex(ValueError, "Original paired image sizes must match"):
                            tools["main"](args)
            self.assertEqual(list((root / "output").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
