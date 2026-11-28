import math
import unittest
from unittest.mock import patch

import torch

from model import TEDFusion_model as geometry


class ClipProbe(torch.nn.Module):
    def encode_text(self, text):
        return text.float().mean(dim=1, keepdim=True).expand(-1, 512)


class HyperbolicGeometryTests(unittest.TestCase):
    def test_distance_from_origin_matches_norm_for_each_curvature(self):
        point = torch.tensor([[0.2, -0.4], [0.3, 0.1]], dtype=torch.float64)
        for curvature in (0.5, 0.7, 1.0):
            with self.subTest(curvature=curvature):
                sqrt_c = math.sqrt(curvature)
                expected = torch.tensor([
                    2 * math.atanh(sqrt_c * math.hypot(*vector)) / sqrt_c
                    for vector in point.tolist()
                ], dtype=point.dtype)
                distance = geometry.poincare_distance(torch.zeros_like(point), point, curvature)
                torch.testing.assert_close(distance, expected)
                torch.testing.assert_close(geometry.hyperbolic_norm(point, curvature), expected)

    def test_nearly_equal_and_identical_points_have_finite_gradients(self):
        left = torch.tensor([[0.1, 0.0], [0.0, 0.0]], requires_grad=True)
        right = torch.tensor([[0.10001, 0.0], [0.0, 0.0]], requires_grad=True)
        distance = geometry.poincare_distance(left, right)
        self.assertGreater(distance[0].item(), 0.0)
        self.assertEqual(distance[1].item(), 0.0)
        distance.sum().backward()
        self.assertTrue(torch.isfinite(left.grad).all())
        self.assertTrue(torch.isfinite(right.grad).all())

    def test_geometry_from_zero_tangent_is_finite_through_backward(self):
        curvature = torch.tensor(0.7)
        tangent = torch.zeros(2, 6, 4, requires_grad=True)
        anchor_tangent = torch.zeros(2, 1, 4, requires_grad=True)
        image = geometry.exp_map(tangent, curvature)
        anchor = geometry.exp_map(anchor_tangent, curvature)
        translated = geometry.mobius_translation(image, anchor, curvature)
        enhanced = geometry.log_map(translated, curvature)
        distance = geometry.poincare_loss(translated, anchor, curvature)
        torch.testing.assert_close(enhanced, torch.zeros_like(enhanced))
        torch.testing.assert_close(distance, torch.zeros_like(distance))
        (enhanced.sum() + distance.sum()).backward()
        self.assertTrue(torch.isfinite(tangent.grad).all())
        self.assertTrue(torch.isfinite(anchor_tangent.grad).all())

    def test_loss_uses_spatial_mean_without_radial_discount(self):
        curvature = torch.tensor(0.7, dtype=torch.float64)
        image = torch.tensor([[[0.1], [0.3]], [[0.5], [0.7]]], dtype=torch.float64)
        anchor = torch.tensor([[[0.2]], [[0.1]]], dtype=torch.float64)
        sqrt_c = math.sqrt(0.7)
        expected = []
        for pixels, text in zip(((0.1, 0.3), (0.5, 0.7)), (0.2, 0.1)):
            pixel_radii = [2 * math.atanh(sqrt_c * value) / sqrt_c for value in pixels]
            anchor_radius = 2 * math.atanh(sqrt_c * text) / sqrt_c
            mean_distance = sum(abs(radius - anchor_radius) for radius in pixel_radii) / len(pixels)
            expected.append(mean_distance)
        torch.testing.assert_close(
            geometry.poincare_loss(image, anchor, curvature),
            torch.tensor(expected, dtype=image.dtype),
        )

    def test_increasing_radial_distance_cannot_discount_its_own_loss(self):
        # The old loss d * exp(-d / 0.5) decreased for d > 0.5. With an
        # origin anchor, exp_map([r, 0]) has geodesic distance exactly 2r.
        for curvature_value in (0.5, 0.7, 1.0):
            with self.subTest(curvature=curvature_value):
                curvature = torch.tensor(curvature_value, dtype=torch.float64)
                radii = torch.tensor([0.1, 0.25, 0.5, 1.0, 2.0, 3.0], dtype=torch.float64)
                tangent = torch.stack([radii, torch.zeros_like(radii)], dim=-1).unsqueeze(1)
                tangent.requires_grad_()
                anchor = torch.zeros(len(radii), 1, 2, dtype=torch.float64)
                loss = geometry.poincare_loss(geometry.exp_map(tangent, curvature), anchor, curvature)
                torch.testing.assert_close(loss, 2 * radii)
                self.assertTrue((loss[1:] > loss[:-1]).all())
                loss.sum().backward()
                self.assertTrue((tangent.grad[:, 0, 0] > 0).all())
                self.assertTrue(torch.isfinite(tangent.grad).all())

                # Increasing just one of four scales cannot be hidden by
                # reducing its weight or the total mass of all scale weights.
                other_scales = torch.full_like(loss, 0.2)
                total = torch.stack((loss, other_scales, other_scales, other_scales)).mean(dim=0)
                torch.testing.assert_close(total, (2 * radii + 0.6) / 4)
                self.assertTrue((total[1:] > total[:-1]).all())

    def test_guidance_maps_channel_vectors_and_reuses_translated_points(self):
        layer = geometry.FeatureWiseAffine(8, 4, curvature=0.7)
        with torch.no_grad():
            layer.image_proj.weight.copy_(torch.eye(4))
            layer.image_proj.bias.zero_()
        image = torch.arange(48, dtype=torch.float32).reshape(2, 4, 2, 3) / 48
        curvature = torch.tensor(0.7)
        anchor = torch.tensor([[[0.1, 0.2]], [[0.0, -0.1]]])
        with (
            patch.object(geometry, "exp_map", wraps=geometry.exp_map) as exp_map,
            patch.object(geometry, "log_map", wraps=geometry.log_map) as log_map,
            patch.object(geometry, "poincare_loss", wraps=geometry.poincare_loss) as loss,
        ):
            enhanced, distance = layer(image, anchor, curvature)
        exp_map.assert_called_once()
        log_map.assert_called_once()
        loss.assert_called_once()
        torch.testing.assert_close(exp_map.call_args.args[0], image.flatten(2).transpose(1, 2))
        self.assertIs(log_map.call_args.args[0], loss.call_args.args[0])
        torch.testing.assert_close(loss.call_args.args[1][..., :2], anchor)
        torch.testing.assert_close(loss.call_args.args[1][..., 2:], torch.zeros_like(anchor))
        self.assertEqual(enhanced.shape, image.shape)
        self.assertEqual(distance.shape, (2,))


class HyperbolicModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(2)

    def setUp(self):
        self.addCleanup(torch.random.set_rng_state, torch.random.get_rng_state())
        torch.manual_seed(21)
        self.a = torch.rand(2, 3, 16, 16)
        self.b = torch.rand_like(self.a)
        self.attributes = torch.rand(2, 8)
        self.text = torch.ones(2, 77, dtype=torch.long)

    def make_model(self, model_class):
        return model_class(ClipProbe(), num_blocks=[1, 1, 1, 1], num_refinement_blocks=1, curvature=0.7)

    def test_training_and_inference_share_one_anchor_across_four_scales(self):
        for model_class in (geometry.TEDFusion, geometry.TEDFusion1):
            with self.subTest(model=model_class.__name__):
                model = self.make_model(model_class).eval()
                captured = []
                handles = [
                    getattr(model, f"prompt_guidance_{level}").register_forward_hook(
                        lambda _layer, inputs, output: captured.append((inputs, output))
                    )
                    for level in range(1, 5)
                ]
                try:
                    with torch.no_grad():
                        if model_class is geometry.TEDFusion:
                            image, _, _, total = model(self.a, self.b, self.text, self.text, self.attributes)
                        else:
                            image = model(self.a, self.b, self.attributes)
                finally:
                    for handle in handles:
                        handle.remove()

                self.assertEqual(len(captured), 4)
                common_anchor = captured[0][0][1]
                self.assertTrue(all(inputs[1] is common_anchor for inputs, _ in captured))
                self.assertEqual(common_anchor.shape, (2, 1, 48))
                for inputs, _ in captured:
                    torch.testing.assert_close(inputs[2], torch.tensor(0.7))
                self.assertEqual(image.shape, self.a.shape)
                self.assertTrue(torch.isfinite(image).all())
                distances = [output[1] for _, output in captured]
                if model_class is geometry.TEDFusion:
                    torch.testing.assert_close(total, torch.stack(distances).mean())
                else:
                    self.assertTrue(all(distance is None for distance in distances))

    def test_dropout_removes_text_from_guidance_only_during_training(self):
        model = self.make_model(geometry.TEDFusion)
        model.drop_prob = 1.0
        with torch.no_grad():
            model.train()
            first = model(self.a, self.b, self.text, self.text, self.attributes)
            second = model(self.a, self.b, self.text * 2, self.text * 3, self.attributes)
            self.assertFalse(torch.equal(first[1], second[1]))
            for index in (0, 2, 3):
                torch.testing.assert_close(first[index], second[index])

            model.eval()
            with patch.object(torch, "rand_like", side_effect=AssertionError("evaluation sampled text dropout")):
                first_eval = model(self.a, self.b, self.text, self.text, self.attributes)
                second_eval = model(self.a, self.b, self.text * 2, self.text * 3, self.attributes)
            self.assertFalse(torch.equal(first_eval[2], second_eval[2]))

    def test_attribute_only_validation_and_deployed_model_are_identical(self):
        training_model = self.make_model(geometry.TEDFusion).eval()
        deployed_model = self.make_model(geometry.TEDFusion1).eval()
        # This is the pre-existing checkpoint difference: the training-only
        # conv48to384 is registered but never used by either image path.
        state = {
            key: value for key, value in training_model.state_dict().items()
            if not key.startswith("conv48to384.")
        }
        incompatible = deployed_model.load_state_dict(state, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        with (
            torch.no_grad(),
            patch.object(ClipProbe, "encode_text", side_effect=AssertionError("deployment used captions")),
            patch.object(torch, "rand_like", side_effect=AssertionError("deployment sampled text dropout")),
        ):
            plain = training_model.forward_inference(self.a, self.b, self.attributes)
            details = training_model.forward_inference(self.a, self.b, self.attributes, return_details=True)
            deployed = deployed_model(self.a, self.b, self.attributes)
            deployed_details = deployed_model.forward_inference(
                self.a, self.b, self.attributes, return_details=True
            )
        torch.testing.assert_close(plain, details[0], rtol=0, atol=0)
        torch.testing.assert_close(plain, deployed, rtol=0, atol=0)
        for first, second in zip(details, deployed_details):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(details[1].shape, (2, 64))
        self.assertEqual(details[2].shape, (2, 64))
        torch.testing.assert_close(details[1], torch.zeros_like(details[1]), rtol=0, atol=0)
        self.assertEqual(details[3].ndim, 0)
        self.assertTrue(torch.isfinite(details[3]))

    def test_full_attribute_dropout_uses_the_same_deployment_operator(self):
        model = self.make_model(geometry.TEDFusion).train()
        model.drop_prob = 1.0
        with torch.no_grad():
            training = model(self.a, self.b, self.text, self.text * 3, self.attributes)
            inference = model.forward_inference(self.a, self.b, self.attributes, return_details=True)
        for index in (0, 2, 3):
            torch.testing.assert_close(training[index], inference[index], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
