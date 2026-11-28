"""Exercise real optimizer/scheduler state restoration, not only LR formulas."""
import copy
import unittest

import torch

from scripts.training_state import (
    EXPERIMENT_PROTOCOL_VERSION, make_schedule_plan, polynomial_lr_factor,
    resolve_schedule_plan, resolve_training_mode, validate_resume_protocol,
)


class ResumeOptimizerTests(unittest.TestCase):
    def make_optimizer(self, value, plan):
        parameter = torch.nn.Parameter(value.clone())
        optimizer = torch.optim.AdamW([parameter], lr=0.03, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: polynomial_lr_factor(step, plan)
        )
        return parameter, optimizer, scheduler

    def step(self, parameter, optimizer, scheduler, target):
        lr = optimizer.param_groups[0]["lr"]
        optimizer.zero_grad()
        (parameter - target).square().sum().backward()
        optimizer.step()
        scheduler.step()
        return lr

    def test_adamw_and_lr_sequence_match_uninterrupted_training(self):
        plan = make_schedule_plan(2, 4, batch_size=1, samples_per_epoch=2)
        parameter, optimizer, scheduler = self.make_optimizer(torch.tensor([1.0, -2.0]), plan)
        for step in range(4):
            self.step(parameter, optimizer, scheduler, step / 10)
        saved = copy.deepcopy({
            "epoch": 1, "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(), "scheduler_plan": plan,
            "parameter": parameter.detach(),
        })
        expected_lrs = [self.step(parameter, optimizer, scheduler, step / 10) for step in range(4, 8)]
        recovered_plan = resolve_schedule_plan(saved, requested_steps_per_epoch=2)
        restored, opt2, schedule2 = self.make_optimizer(saved["parameter"], recovered_plan)
        opt2.load_state_dict(saved["optimizer"])
        schedule2.load_state_dict(saved["lr_scheduler"])
        actual_lrs = [self.step(restored, opt2, schedule2, step / 10) for step in range(4, 8)]
        self.assertEqual(actual_lrs, expected_lrs)
        torch.testing.assert_close(restored, parameter, rtol=0, atol=0)
        self.assertEqual(opt2.param_groups[0]["lr"], 0.0)

    def test_freezing_policy_survives_two_resumed_checkpoints(self):
        state = resolve_training_mode(initial_weights=True)
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=0.01)
        for _ in range(2):
            # args.weights is deliberately empty, as after the first resume.
            saved = {"args": {"weights": "", "resume": "previous.pth"},
                     "training_state": copy.deepcopy(state), "optimizer": optimizer.state_dict()}
            state = resolve_training_mode(saved, optimizer_param_counts={"frozen": 1, "unfrozen": 5})
            self.assertTrue(state["fine_tuning"])
            self.assertTrue(state["freeze_upsamplers"])
            reloaded = torch.optim.AdamW([parameter], lr=0.01)
            reloaded.load_state_dict(saved["optimizer"])
            optimizer = reloaded

    def test_old_experiment_cannot_silently_resume_under_new_objective(self):
        for checkpoint in ({}, {"experiment_protocol_version": 1}):
            with self.assertRaisesRegex(ValueError, "--weights"):
                validate_resume_protocol(checkpoint)
        validate_resume_protocol({"experiment_protocol_version": EXPERIMENT_PROTOCOL_VERSION})


if __name__ == "__main__":
    unittest.main()
