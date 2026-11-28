"""Resume regressions that run without loading PyTorch, BLIP, or CLIP."""

import argparse
import copy
import math
import unittest

from scripts.training_state import (
    make_schedule_plan, polynomial_lr_factor,
    resolve_schedule_plan, resolve_training_mode,
)


COUNTS = {"frozen": 8, "unfrozen": 12}


def checkpoint_arguments(**overrides):
    values = dict(weights="", resume="", train_upsamplers=False,
                  epochs=25, batch_size=2, samples_per_epoch=800)
    values.update(overrides)
    return argparse.Namespace(**values)


def checkpoint(frozen=True, **argument_overrides):
    return {
        "args": checkpoint_arguments(**argument_overrides),
        "optimizer": {"param_groups": [{"params": list(range(8 if frozen else 12))}]},
        "epoch": 2,
        "lr_scheduler": {"last_epoch": 1200, "_step_count": 1201},
    }


class TrainingModeTests(unittest.TestCase):
    def test_new_training_modes(self):
        self.assertEqual(resolve_training_mode()["fine_tuning"], False)
        self.assertEqual(resolve_training_mode()["freeze_upsamplers"], False)
        self.assertEqual(resolve_training_mode(initial_weights=True)["freeze_upsamplers"], True)
        self.assertEqual(resolve_training_mode(initial_weights=True, train_upsamplers=True)["freeze_upsamplers"], False)

    def test_freezing_survives_two_resumes_without_original_weights_argument(self):
        saved = checkpoint(weights="pretrained.pth")
        first_mode = resolve_training_mode(saved, optimizer_param_counts=COUNTS)
        resumed = checkpoint(resume="first_checkpoint.pth")
        resumed["training_state"] = copy.deepcopy(first_mode)
        second_mode = resolve_training_mode(resumed, optimizer_param_counts=COUNTS)
        self.assertEqual(first_mode, second_mode)
        self.assertTrue(second_mode["fine_tuning"])
        self.assertTrue(second_mode["freeze_upsamplers"])

    def test_legacy_second_resume_recovers_freezing_from_optimizer(self):
        saved = checkpoint(resume="first_checkpoint.pth")
        recovered = resolve_training_mode(saved, optimizer_param_counts=COUNTS)
        self.assertTrue(recovered["fine_tuning"])
        self.assertTrue(recovered["freeze_upsamplers"])

    def test_unfrozen_legacy_resume_without_history_is_ambiguous(self):
        saved = checkpoint(frozen=False, resume="first_checkpoint.pth")
        with self.assertRaisesRegex(ValueError, "cannot be recovered unambiguously"):
            resolve_training_mode(saved, optimizer_param_counts=COUNTS)

    def test_scratch_and_unfrozen_finetuning_are_distinguished_by_saved_history(self):
        scratch = resolve_training_mode(checkpoint(frozen=False), optimizer_param_counts=COUNTS)
        fine = resolve_training_mode(checkpoint(frozen=False, weights="a.pth", train_upsamplers=True),
                                     optimizer_param_counts=COUNTS)
        self.assertFalse(scratch["fine_tuning"])
        self.assertTrue(fine["fine_tuning"])
        self.assertFalse(fine["freeze_upsamplers"])

    def test_cli_and_saved_optimizer_conflicts_fail_before_loading(self):
        saved = checkpoint(weights="pretrained.pth")
        saved["training_state"] = resolve_training_mode(saved, optimizer_param_counts=COUNTS)
        with self.assertRaisesRegex(ValueError, "--weights"):
            resolve_training_mode(saved, train_upsamplers=True, optimizer_param_counts=COUNTS)
        saved["training_state"]["freeze_upsamplers"] = False
        with self.assertRaisesRegex(ValueError, "optimizer layout"):
            resolve_training_mode(saved, optimizer_param_counts=COUNTS)

    def test_legacy_args_optimizer_conflict_and_missing_history_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "optimizer layout"):
            resolve_training_mode(checkpoint(frozen=False, weights="pretrained.pth"),
                                  optimizer_param_counts=COUNTS)
        with self.assertRaisesRegex(ValueError, "cannot be recovered unambiguously"):
            resolve_training_mode({})
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            resolve_training_mode(checkpoint(), initial_weights=True)


class SchedulePlanTests(unittest.TestCase):
    def test_explicit_plan_round_trip_preserves_horizon_and_warmup(self):
        saved = checkpoint()
        saved["scheduler_plan"] = make_schedule_plan(400, 25, warmup_epochs=2,
                                                     warmup_factor=0.02, batch_size=2,
                                                     samples_per_epoch=800)
        self.assertEqual(resolve_schedule_plan(saved), saved["scheduler_plan"])
        self.assertEqual(resolve_schedule_plan(saved, 400, 25, 2, 800), saved["scheduler_plan"])

    def test_resuming_rejects_changed_steps_epochs_batch_or_samples(self):
        saved = checkpoint()
        saved["scheduler_plan"] = make_schedule_plan(400, 25, batch_size=2, samples_per_epoch=800)
        cases = ({"requested_steps_per_epoch": 200}, {"requested_epochs": 15},
                 {"requested_batch_size": 4}, {"requested_samples_per_epoch": 400},
                 {"requested_samples_per_epoch": 799})
        for case in cases:
            with self.subTest(case=case), self.assertRaisesRegex(ValueError, "--weights"):
                resolve_schedule_plan(saved, **case)

    def test_legacy_plan_is_recovered_without_applying_cli_defaults(self):
        plan = resolve_schedule_plan(checkpoint())
        self.assertEqual(plan, make_schedule_plan(400, 25, batch_size=2, samples_per_epoch=800))
        self.assertEqual(plan["total_epochs"], 25)
        with self.assertRaisesRegex(ValueError, "total_epochs=25"):
            resolve_schedule_plan(checkpoint(), requested_epochs=15)

    def test_legacy_full_dataset_uses_observed_step_count(self):
        saved = checkpoint(samples_per_epoch=0)
        saved["lr_scheduler"]["last_epoch"] = 1464  # ceil(975 / 2) * 3
        plan = resolve_schedule_plan(saved)
        self.assertEqual(plan["steps_per_epoch"], 488)
        with self.assertRaisesRegex(ValueError, "steps_per_epoch=488"):
            resolve_schedule_plan(saved, requested_steps_per_epoch=400)

    def test_inconsistent_or_insufficient_history_is_rejected(self):
        saved = checkpoint()
        saved["lr_scheduler"]["last_epoch"] = 600
        with self.assertRaisesRegex(ValueError, "arguments disagree"):
            resolve_schedule_plan(saved)
        with self.assertRaisesRegex(ValueError, "original total epoch count"):
            resolve_schedule_plan({})
        saved = {"args": {"epochs": 25, "samples_per_epoch": 0, "batch_size": 2}}
        with self.assertRaisesRegex(ValueError, "insufficient information"):
            resolve_schedule_plan(saved)

    def test_explicit_plan_progress_must_match_saved_steps(self):
        saved = checkpoint()
        saved["scheduler_plan"] = make_schedule_plan(200, 25)
        with self.assertRaisesRegex(ValueError, "progress disagrees"):
            resolve_schedule_plan(saved)
        saved["lr_scheduler"]["last_epoch"] = 1201
        with self.assertRaisesRegex(ValueError, "epoch-boundary"):
            resolve_schedule_plan(saved)

    def test_lr_sequence_is_identical_across_resume(self):
        saved = checkpoint()
        plan = make_schedule_plan(400, 25, batch_size=2, samples_per_epoch=800)
        saved["scheduler_plan"] = plan
        restored = resolve_schedule_plan(saved, requested_steps_per_epoch=400)
        for step in (1200, 1201, 7999, 9999, 10000, 10001, 100000):
            self.assertEqual(polynomial_lr_factor(step, plan), polynomial_lr_factor(step, restored))

    def test_warmup_decay_and_out_of_bounds_never_produce_complex_or_negative_lr(self):
        plan = make_schedule_plan(4, 5)
        self.assertEqual(polynomial_lr_factor(0, plan), 0.001)
        self.assertEqual(polynomial_lr_factor(4, plan), 1.0)
        for step in range(5, 25):
            factor = polynomial_lr_factor(step, plan)
            self.assertIsInstance(factor, float)
            self.assertTrue(math.isfinite(factor))
            self.assertGreaterEqual(factor, 0.0)
        self.assertEqual(polynomial_lr_factor(20, plan), 0.0)
        self.assertEqual(polynomial_lr_factor(21, plan), 0.0)
        no_warmup = make_schedule_plan(4, 5, warmup=False)
        self.assertEqual(polynomial_lr_factor(0, no_warmup), 1.0)
        one_epoch = make_schedule_plan(4, 1)
        self.assertEqual(polynomial_lr_factor(4, one_epoch), 0.0)
        with self.assertRaisesRegex(ValueError, "scheduler step"):
            polynomial_lr_factor(-1, plan)


if __name__ == "__main__":
    unittest.main()
