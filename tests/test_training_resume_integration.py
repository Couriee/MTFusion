"""Run the real training entry point through two interrupted CPU resumes."""

import argparse
import contextlib
import datetime
import io
import os
import random
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

import train_fusion as training
from scripts.training_state import EXPERIMENT_PROTOCOL_VERSION, polynomial_lr_factor


class _EndPhase(Exception):
    """Simulate an interruption before the next epoch updates any parameters."""


class _SilentWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


class _Upsampler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Match the real upsampler's body[0] and four PixelShuffle phases.
        self.body = torch.nn.Sequential(torch.nn.Conv2d(1, 4, 1, bias=False))


class _FakeModel(torch.nn.Module):
    def __init__(self, model_clip, curvature=1.0):
        super().__init__()
        self.model_clip = model_clip
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
        for name in training.UPSAMPLE_MODULE_NAMES:
            setattr(self, name, _Upsampler())

    def forward(self, visible):
        return visible * self.scale


def _arguments(**overrides):
    values = dict(
        epochs=None, batch_size=None, lr=None, weight_decay=None,
        grad_clip_norm=None, samples_per_epoch=None, max_val_samples=None,
        curvature=None, dataset_path=None, low_light_path=None,
        over_exposure_path=None, ir_low_contrast_path=None, ir_noise_path=None,
        weights="", train_upsamplers=None, val_every_epcho=2, resume="",
        use_dp=False, device="cpu", gpu_id="0",
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class TrainingResumeIntegrationTests(unittest.TestCase):
    def test_main_preserves_freezing_optimizer_and_plan_across_two_resumes(self):
        original_directory = Path.cwd()
        original_python_rng = random.getstate()
        original_numpy_rng = np.random.get_state()
        original_torch_rng = torch.get_rng_state()
        original_determinism = torch.are_deterministic_algorithms_enabled()
        original_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        original_cudnn = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
        observed_steps = []
        observed_lrs = []
        observed_clips = []
        created_models = []
        stop_before_epoch = [3]

        def make_model(model_clip, **kwargs):
            model = _FakeModel(model_clip, **kwargs)
            created_models.append(model)
            return model

        def train_epoch(model, optimizer, data_loader, lr_scheduler, device,
                        epoch, grad_clip_norm, **kwargs):
            if epoch == stop_before_epoch[0]:
                raise _EndPhase()
            self.assertEqual(data_loader.batch_size, 1)
            self.assertEqual(len(data_loader), 1)
            self.assertEqual(len(data_loader.dataset), 1)
            self.assertEqual(grad_clip_norm, 0.25)
            observed_clips.append(grad_clip_norm)
            self.assertEqual(len(optimizer.param_groups[0]["params"]), 1)
            self.assertIs(optimizer.param_groups[0]["params"][0], model.scale)
            self.assertFalse(any(p.requires_grad for p in model.model_clip.parameters()))
            for name in training.UPSAMPLE_MODULE_NAMES:
                self.assertFalse(any(p.requires_grad for p in getattr(model, name).parameters()))

            # A resumed AdamW must arrive with the real saved moments and step,
            # rather than merely recreating a scheduler with the same formula.
            before = optimizer.state.get(model.scale, {}).get("step", 0)
            self.assertEqual(float(before), epoch)
            if epoch:
                self.assertIn("exp_avg", optimizer.state[model.scale])
                self.assertGreater(float(optimizer.state[model.scale]["exp_avg"].abs()), 0.0)
            observed_steps.append((epoch, int(float(before))))
            model.train()
            for batch in data_loader:
                observed_lrs.append(optimizer.param_groups[0]["lr"])
                optimizer.zero_grad(set_to_none=True)
                loss = (model(batch[0].to(device)) - 0.1).square().mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                lr_scheduler.step()
            self.assertEqual(float(optimizer.state[model.scale]["step"]), epoch + 1)
            return (float(loss.detach()), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    optimizer.param_groups[0]["lr"])

        def evaluate_epoch(data_loader, **kwargs):
            self.assertEqual(len(data_loader), 1)
            return (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        try:
            with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
                root = Path(directory)
                dataset = root / "dataset"
                for phase in ("train", "eval"):
                    for modality in ("Visible", "Infrared"):
                        folder = dataset / phase / modality
                        folder.mkdir(parents=True)
                        pixels = np.full((96, 96, 3), 204 if modality == "Visible" else 128,
                                         dtype=np.uint8)
                        Image.fromarray(pixels).save(folder / "sample.png")
                initial = _FakeModel(torch.nn.Linear(1, 1, bias=False))
                initial_state = {name: tensor.clone() for name, tensor in initial.state_dict().items()}
                initial_path = root / "initial.pth"
                torch.save({"model": initial_state, "curvature": 1.0}, initial_path)

                stack.enter_context(torch.random.fork_rng(devices=[]))
                stack.enter_context(patch.dict(os.environ))
                stack.enter_context(patch.object(torch.cuda, "is_available", return_value=False))
                stack.enter_context(patch.object(training.clip, "load", side_effect=lambda *a, **k:
                                                (torch.nn.Linear(1, 1, bias=False), None)))
                stack.enter_context(patch.object(training, "create_model", side_effect=make_model))
                stack.enter_context(patch.object(training, "SummaryWriter", _SilentWriter))
                stack.enter_context(patch.object(training, "train_one_epoch", side_effect=train_epoch))
                stack.enter_context(patch.object(training, "evaluate", side_effect=evaluate_epoch))
                # Relative experiment paths stay inside the temporary directory.
                os.chdir(root)
                stack.callback(os.chdir, original_directory)

                def run(arguments, run_index, interrupted):
                    stamp = datetime.datetime(2026, 1, 1, 0, 0, run_index)
                    clock = types.SimpleNamespace(datetime=types.SimpleNamespace(now=lambda: stamp))
                    # Replace only this module's clock, leaving datetime itself
                    # intact for PyTorch and its lazily imported dependencies.
                    with patch.object(training, "datetime", clock), contextlib.redirect_stdout(io.StringIO()):
                        if interrupted:
                            with self.assertRaises(_EndPhase):
                                training.main(arguments)
                        else:
                            training.main(arguments)
                    path = (root / "experiments" / ("TEDFusion_train_" + stamp.strftime("%Y%m%d-%H%M%S"))
                            / "weights" / "checkpoint_lastest.pth")
                    self.assertTrue(path.is_file())
                    return path, torch.load(path, map_location="cpu", weights_only=False)

                first_path, first = run(_arguments(
                    weights=str(initial_path), dataset_path=str(dataset), epochs=7,
                    batch_size=1, samples_per_epoch=1, grad_clip_norm=0.25,
                ), 1, interrupted=True)
                self.assertEqual(first["epoch"], 2)
                self.assertEqual(first["lr_scheduler"]["last_epoch"], 3)

                stop_before_epoch[0] = 5
                # Only --resume is supplied; the CLI's critical options remain
                # None and must be restored from the checkpoint by main().
                second_args = _arguments(resume=str(first_path))
                second_path, second = run(second_args, 2, interrupted=True)
                self.assertEqual(second["epoch"], 4)
                self.assertEqual(second["lr_scheduler"]["last_epoch"], 5)
                self.assertEqual(second["args"].weights, "")

                stop_before_epoch[0] = None
                final_args = _arguments(resume=str(second_path))
                _, final = run(final_args, 3, interrupted=False)
                self.assertEqual(final["epoch"], 6)
                self.assertEqual(final["lr_scheduler"]["last_epoch"], 7)
                self.assertEqual(final["optimizer"]["param_groups"][0]["lr"], 0.0)

                for saved in (first, second, final):
                    self.assertEqual(saved["training_state"], {
                        "version": 1, "fine_tuning": True, "freeze_upsamplers": True,
                    })
                    self.assertEqual(saved["experiment_protocol_version"], EXPERIMENT_PROTOCOL_VERSION)
                    self.assertEqual(saved["scheduler_plan"]["steps_per_epoch"], 1)
                    self.assertEqual(saved["scheduler_plan"]["total_epochs"], 7)
                    self.assertEqual(saved["scheduler_plan"]["batch_size"], 1)
                    self.assertEqual(saved["scheduler_plan"]["samples_per_epoch"], 1)
                    self.assertEqual(saved["args"].grad_clip_norm, 0.25)
                    self.assertEqual(len(saved["optimizer"]["param_groups"][0]["params"]), 1)
                    for name, tensor in saved["model"].items():
                        if name.startswith(tuple(module + "." for module in training.UPSAMPLE_MODULE_NAMES)):
                            torch.testing.assert_close(tensor, initial_state[name], rtol=0, atol=0)
                for arguments in (second_args, final_args):
                    self.assertEqual((arguments.epochs, arguments.batch_size, arguments.samples_per_epoch),
                                     (7, 1, 1))
                    self.assertEqual(arguments.grad_clip_norm, 0.25)
                    self.assertEqual(arguments.dataset_path, str(dataset))
                self.assertEqual(observed_steps, [(epoch, epoch) for epoch in range(7)])
                self.assertEqual(observed_clips, [0.25] * 7)
                self.assertEqual(len(created_models), 3)
                self.assertEqual(observed_lrs, [2e-5 * polynomial_lr_factor(step, final["scheduler_plan"])
                                               for step in range(7)])
                torch.testing.assert_close(created_models[1].scale.detach(), second["model"]["scale"],
                                           rtol=0, atol=0)
                self.assertLess(float(final["model"]["scale"]), float(initial_state["scale"]))
        finally:
            os.chdir(original_directory)
            random.setstate(original_python_rng)
            np.random.set_state(original_numpy_rng)
            torch.set_rng_state(original_torch_rng)
            torch.use_deterministic_algorithms(original_determinism, warn_only=original_warn_only)
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = original_cudnn


if __name__ == "__main__":
    unittest.main()
