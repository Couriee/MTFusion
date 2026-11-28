"""Coverage and image-preservation checks for model selection."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.validation import select_validation_samples


class ValidationCoverageTests(unittest.TestCase):
    def test_default_covers_all_108_msrs_pairs(self):
        result = select_validation_samples(["msrs"], {"msrs": 108})
        self.assertEqual(result, [("msrs", i) for i in range(108)])

    def test_cap_covers_all_tasks_and_full_index_ranges(self):
        keys = ["night", "exposure", "contrast", "noise"]
        lengths = dict.fromkeys(keys, 108)
        result = select_validation_samples(keys, lengths, 80)
        self.assertEqual(len(result), 80)
        self.assertEqual(len(set(result)), 80)
        for key in keys:
            indices = [index for task, index in result if task == key]
            self.assertEqual(len(indices), 20)
            self.assertLess(indices[0], 6)
            self.assertGreater(indices[-1], 100)
        self.assertEqual(result, select_validation_samples(keys, lengths, 80))

    def test_small_tasks_release_their_unused_budget(self):
        result = select_validation_samples(["small", "large"], {"small": 1, "large": 100}, 8)
        self.assertEqual(len(result), 8)
        self.assertIn(("small", 0), result)
        self.assertEqual(len(set(result)), 8)

    def test_cap_cannot_silently_skip_entire_tasks(self):
        with self.assertRaisesRegex(ValueError, "number of validation tasks"):
            select_validation_samples(["a", "b"], {"a": 10, "b": 10}, 1)


class FullImageValidationTests(unittest.TestCase):
    def test_real_validation_uses_attributes_full_image_and_absolute_output(self):
        import numpy as np
        import torch
        from PIL import Image
        from scripts import utils

        class DeploymentProbe(torch.nn.Module):
            def forward(self, *args, **kwargs):
                raise AssertionError("Validation used the text-training forward path")

            def forward_inference(self, visible, infrared, feature, return_details=False):
                self.observed_shape = tuple(visible.shape)
                self.observed_feature = feature.clone()
                if not return_details:
                    raise AssertionError("Validation did not request the deployment loss details")
                placeholder = visible.new_zeros((visible.shape[0], 64))
                return visible, placeholder, placeholder, visible.new_zeros(())

        visible = torch.full((1, 3, 33, 49), 0.4)
        infrared = torch.full_like(visible, 0.6)
        batch = (visible, infrared, visible.clone(), infrared.clone(), visible.clone(), ("msrs",), ("sample",))
        model = DeploymentProbe()
        with TemporaryDirectory() as folder, patch.object(
            utils, "load_local_blip", side_effect=AssertionError("Validation tried to load caption weights")
        ):
            losses = utils.evaluate(model, [batch], "cpu", 2, 0.001, folder)
            self.assertEqual(model.observed_shape, (1, 3, 48, 64))
            torch.testing.assert_close(model.observed_feature[:, [1, 5]], torch.tensor([[0.4, 0.6]]))
            self.assertTrue(all(np.isfinite(value) for value in losses))
            with Image.open(Path(folder) / "2" / "sample.png") as output:
                self.assertEqual(output.size, (49, 33))
                np.testing.assert_array_equal(np.asarray(output), np.full((33, 49, 3), 102, dtype=np.uint8))

    def test_validation_transform_preserves_original_pixels_and_extent(self):
        import ast
        import numpy as np
        from PIL import Image
        import transforms as T

        path = Path(__file__).resolve().parents[1] / "train_fusion.py"
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        assignment = next(node for node in main.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "data_transform" for target in node.targets))
        scope = {"T": T}
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(path), "exec"), scope)
        pixels = np.arange(33 * 49 * 3, dtype=np.uint8).reshape(33, 49, 3)
        image = Image.fromarray(pixels)
        transformed = scope["data_transform"]["val"](*(image.copy() for _ in range(5)))
        self.assertTrue(all(tuple(value.shape) == (3, 33, 49) for value in transformed))
        np.testing.assert_array_equal(
            np.rint(transformed[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8), pixels
        )


if __name__ == "__main__":
    unittest.main()
