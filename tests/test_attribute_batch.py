"""Regression checks for per-image attributes and their model input boundary.

Run from MTFusion: python -B -m unittest discover -s tests -v
"""

import ast
import copy
from pathlib import Path
import unittest
import warnings
from unittest.mock import patch

import cv2
import numpy as np
import torch
from skimage.feature import graycomatrix

from args_fusion import args
from model.TEDFusion_model import TEDFusion, TEDFusion1


def _load_attribute_tools(relative_path):
    """Load the real local helpers without loading BLIP or caption weights."""
    path = Path(__file__).resolve().parents[1] / relative_path
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    names = {
        '_check_rgb_batch', 'mergy_RGB_to_HSV', 'rgb_to_gray',
        'calculate_contrast', 'calculate_edge', '_texture_complexity',
    }
    namespace = {
        'np': np, 'torch': torch, 'cv2': cv2,
        'graycomatrix': graycomatrix,
    }
    helpers = ast.Module(body=[
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ], type_ignores=[])
    exec(compile(helpers, str(path), 'exec'), namespace)
    return tree, namespace


def _stage_extractor(tree, namespace, function_name):
    """Execute the attribute statements actually used by a stage's loop."""
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
    model_call = next(node for node in ast.walk(function)
                      if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                      and node.func.id == 'model')
    # Bind images in the order actually passed to the model. Normalizing the
    # attribute variable names alone would hide a reversed VI/IR order.
    image_names = tuple(node.id for node in model_call.args[:2])
    loop = next(node for node in function.body if isinstance(node, ast.For))
    start = next(i for i, node in enumerate(loop.body)
                 if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                 and isinstance(node.value.func, ast.Name) and node.value.func.id == '_check_rgb_batch')
    stop = next(i for i, node in enumerate(loop.body[start:], start)
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'as_tensor')
    wrapper = ast.parse('def extract({}):\n    pass'.format(', '.join(image_names))).body[0]
    wrapper.body = copy.deepcopy(loop.body[start:stop + 1]) + [ast.Return(value=ast.Name(id='feature', ctx=ast.Load()))]
    module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
    exec(compile(module, function_name, 'exec'), namespace)
    return namespace['extract']


_utils_tree, _utils_tools = _load_attribute_tools('scripts/utils.py')
_test_tree, _test_tools = _load_attribute_tools('test_from_dataset.py')
STAGE_EXTRACTORS = {
    'train': _stage_extractor(_utils_tree, _utils_tools, 'train_one_epoch'),
    'validation': _stage_extractor(_utils_tree, _utils_tools, 'evaluate'),
    'test': _stage_extractor(_test_tree, _test_tools, 'main'),
}
calculate_contrast = _utils_tools['calculate_contrast']
calculate_edge = _utils_tools['calculate_edge']
mergy_RGB_to_HSV = _utils_tools['mergy_RGB_to_HSV']
extract_pair_attributes = STAGE_EXTRACTORS['train']


class ClipProbe(torch.nn.Module):
    """Deterministic text embeddings; no pretrained model download is needed."""

    def encode_text(self, text):
        return text.float().mean(dim=1, keepdim=True).expand(-1, 512)


class AttributeBatchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(122345)

    def test_hsv_keeps_both_images(self):
        images = torch.full((2, 3, 8, 8), 0.2)
        images[1, 0] = 0.8
        images[1, 1:] = 0.1
        h, s, v = mergy_RGB_to_HSV(images)
        self.assertEqual(h.shape, (2, 8, 8))
        np.testing.assert_allclose(s.mean(axis=(1, 2)), [0, 0.875], atol=1e-6)
        np.testing.assert_allclose(v.mean(axis=(1, 2)), [0.2, 0.8], atol=1e-6)

    def test_constant_images_have_no_spatial_edge(self):
        gray = torch.stack([torch.full((3, 8, 8), 0.2), torch.full((3, 8, 8), 0.8)])
        red = torch.zeros(1, 3, 8, 8)
        red[:, 0] = 1
        for images in (gray, gray[:1], red):
            np.testing.assert_array_equal(calculate_edge(images), np.zeros(images.shape[0]))

    def test_spatial_ramps_have_expected_edge(self):
        horizontal = torch.arange(8, dtype=torch.float32).view(1, 1, 8).expand(3, 8, 8)
        vertical = (2 * torch.arange(8, dtype=torch.float32)).view(1, 8, 1).expand(3, 8, 8)
        np.testing.assert_allclose(calculate_edge(torch.stack([horizontal, vertical])), [1, 2])

    def test_all_attributes_are_independent_of_batch_companions(self):
        for batch in (1, 2, 4):
            with self.subTest(batch=batch):
                a, b = torch.rand(batch, 3, 8, 16), torch.rand(batch, 3, 8, 16)
                combined = extract_pair_attributes(a, b)
                separate = torch.cat([
                    extract_pair_attributes(a[i:i + 1], b[i:i + 1])
                    for i in range(batch)
                ])
                self.assertEqual(combined.shape, (batch, 8))
                torch.testing.assert_close(combined, separate, rtol=1e-6, atol=1e-7)
                order = torch.arange(batch - 1, -1, -1)
                torch.testing.assert_close(extract_pair_attributes(a[order], b[order]), combined[order])

    def test_attribute_columns_keep_existing_order(self):
        a, b = torch.full((1, 3, 8, 16), 0.2), torch.zeros(1, 3, 8, 16)
        b[:, 0] = 0.8
        b[:, 1:] = 0.1
        feature = extract_pair_attributes(a, b)
        torch.testing.assert_close(feature[0, [0, 1, 4, 5]], torch.tensor([0, 0.2, 0.875, 0.8]))
        torch.testing.assert_close(feature[:, 3], calculate_contrast(a))
        torch.testing.assert_close(feature[:, 7], calculate_contrast(b))

    def test_unpaired_shapes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Paired image shapes'):
            extract_pair_attributes(torch.rand(2, 3, 8, 8), torch.rand(1, 3, 8, 8))

    def test_texture_matches_appendix_a_for_alternating_rows(self):
        rows = (torch.arange(32) % 2) * (4.0 / 255)
        image = rows.view(1, 1, 32, 1).expand(1, 3, 32, 32)
        # Each matrix has two equally probable entries, so E = 0.5.
        # Six diagonal offsets cross rows of opposite gray levels, so
        # H = (10 * 1 + 6 * (1 / (1 + 4))) / 16 = 0.7.
        expected = torch.tensor([[0.58, 0.56]])  # Visible, infrared
        for name, extractor in STAGE_EXTRACTORS.items():
            with self.subTest(stage=name):
                torch.testing.assert_close(extractor(image, image)[:, [2, 6]], expected)

    def test_independent_stage_sections_produce_the_same_repaired_attributes(self):
        a, b = torch.rand(4, 3, 8, 16), torch.rand(4, 3, 8, 16)
        expected = STAGE_EXTRACTORS['train'](a, b)
        for name, extractor in STAGE_EXTRACTORS.items():
            with self.subTest(stage=name):
                torch.testing.assert_close(extractor(a, b), expected, rtol=0, atol=0)
                separate = torch.cat([extractor(a[i:i + 1], b[i:i + 1]) for i in range(4)])
                torch.testing.assert_close(extractor(a, b), separate, rtol=0, atol=0)

    def test_inference_attributes_match_visible_then_infrared_inputs(self):
        visible = torch.full((2, 3, 8, 16), 0.1)
        visible[:, 0] = torch.linspace(0.6, 0.95, 16)
        visible[:, 2] = 0.2
        infrared = torch.rand(2, 1, 8, 16).expand(-1, 3, -1, -1).clone()
        expected = STAGE_EXTRACTORS['train'](visible, infrared)
        actual = STAGE_EXTRACTORS['test'](visible, infrared)
        # Check all eight fields, including the position-specific texture
        # weights; merely swapping two precomputed groups is insufficient.
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue((actual[:, 0] > 0.5).all())
        torch.testing.assert_close(actual[:, 4], torch.zeros(2), rtol=0, atol=0)


class ModelAttributeBoundaryTests(unittest.TestCase):
    def test_training_and_inference_models_accept_per_image_attributes(self):
        # Exercise the actual encoders, gates, TIAM and decoders at reduced depth.
        # Keep all channel widths and attribute projection dimensions unchanged.
        torch.set_num_threads(2)
        with patch.object(args, 'device', 'cpu'), warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            for model_class in (TEDFusion, TEDFusion1):
                with self.subTest(model=model_class.__name__):
                    torch.manual_seed(122345)
                    model = model_class(ClipProbe(), num_blocks=[1, 1, 1, 1], num_refinement_blocks=1)
                    model.eval()
                    # Disable the existing custom stochastic branch for the
                    # compatibility comparison; its implementation is unchanged.
                    model.is_training = 0
                    for batch in (1, 2):
                        a, b = torch.rand(batch, 3, 16, 16), torch.rand(batch, 3, 16, 16)
                        feature = extract_pair_attributes(a, b)
                        text1 = torch.ones(batch, 77, dtype=torch.long)
                        text2 = torch.zeros_like(text1)
                        captured = []
                        handle = model.attr_proj1.register_forward_pre_hook(
                            lambda _module, inputs: captured.append(inputs[0].detach().clone())
                        )

                        def forward(attributes):
                            if model_class is TEDFusion:
                                return model(a, b, text1, text2, attributes)
                            return model(a, b, attributes)

                        with torch.no_grad():
                            result = forward(feature)
                            image = result[0] if model_class is TEDFusion else result
                            self.assertEqual(image.shape, a.shape)
                            self.assertTrue(torch.isfinite(image).all())
                            torch.testing.assert_close(captured[0], feature[:, :2])
                            if batch == 1:
                                legacy_result = forward(feature[0])
                                legacy_image = legacy_result[0] if model_class is TEDFusion else legacy_result
                                torch.testing.assert_close(image, legacy_image)
                            else:
                                with self.assertRaisesRegex(ValueError, 'per-image attributes'):
                                    forward(feature[0])
                            with self.assertRaisesRegex(ValueError, 'per-image attributes'):
                                forward(feature.unsqueeze(0))
                        handle.remove()

                        if model_class is TEDFusion and batch == 2:
                            model.train()
                            model.is_training = 1
                            model.zero_grad(set_to_none=True)
                            image, _, _, distance = forward(feature)
                            loss = image.square().mean() + 0.005 * distance
                            self.assertTrue(torch.isfinite(loss))
                            loss.backward()
                            for projection in (model.attr_proj1, model.attr_proj2, model.attr_proj3, model.attr_proj4):
                                self.assertIsNotNone(projection.weight.grad)
                                self.assertTrue(torch.isfinite(projection.weight.grad).all())


if __name__ == '__main__':
    unittest.main()
