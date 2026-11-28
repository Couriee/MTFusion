"""Serializable resume policy and LR plans, independent of PyTorch.

``--resume`` continues one experiment. Changing its trainable parameters or LR
horizon requires starting a new experiment with ``--weights`` instead.
"""

import math
from collections.abc import Mapping
from numbers import Integral, Real


_NEW_EXPERIMENT = "Use --weights to start a new experiment."
EXPERIMENT_PROTOCOL_VERSION = 2


def validate_resume_protocol(checkpoint):
    """An optimizer resume must not silently switch training/evaluation targets."""
    if checkpoint.get("experiment_protocol_version") != EXPERIMENT_PROTOCOL_VERSION:
        _fail("Checkpoint predates the unified TIAM, distance loss and full-image validation protocol.")


def _fail(message):
    raise ValueError("{} {}".format(message, _NEW_EXPERIMENT))


def _argument(arguments, name, default=None):
    if isinstance(arguments, Mapping):
        return arguments.get(name, default)
    return getattr(arguments, name, default)


def _integer(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, Integral):
        _fail("{} must be an integer, got {!r}.".format(name, value))
    value = int(value)
    if minimum is not None and value < minimum:
        _fail("{} must be >= {}, got {}.".format(name, minimum, value))
    return value


def _real(value, name, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        _fail("{} must be finite, got {!r}.".format(name, value))
    value = float(value)
    if ((minimum is not None and value < minimum)
            or (maximum is not None and value > maximum)):
        _fail("{} is outside the supported range.".format(name))
    return value


def _boolean(value, name):
    if not isinstance(value, bool):
        _fail("{} must be a boolean, got {!r}.".format(name, value))
    return value


def _optimizer_freeze_state(checkpoint, parameter_counts):
    """Infer only the two optimizer layouts this trainer actually constructs."""
    if parameter_counts is None:
        return None
    if not isinstance(parameter_counts, Mapping):
        _fail("optimizer_param_counts must map frozen/unfrozen to parameter counts.")
    expected = {
        key: _integer(parameter_counts.get(key), key + " parameter count", 1)
        for key in ("frozen", "unfrozen")
    }
    if expected["frozen"] == expected["unfrozen"]:
        _fail("Optimizer parameter counts do not distinguish the upsampler states.")
    optimizer = checkpoint.get("optimizer")
    if not isinstance(optimizer, Mapping) or not optimizer.get("param_groups"):
        _fail("Checkpoint has no optimizer parameter groups to verify freezing.")
    groups = optimizer["param_groups"]
    try:
        saved_count = sum(len(group["params"]) for group in groups)
    except (KeyError, TypeError):
        _fail("Checkpoint optimizer parameter groups are invalid.")
    matches = [key for key, count in expected.items() if count == saved_count]
    if len(matches) != 1:
        _fail("Checkpoint optimizer has {} parameters; expected {} frozen or {} unfrozen."
              .format(saved_count, expected["frozen"], expected["unfrozen"]))
    return matches[0] == "frozen"


def resolve_training_mode(checkpoint=None, initial_weights=False,
                          train_upsamplers=None, optimizer_param_counts=None):
    """Return the actual mode to persist as ``checkpoint['training_state']``.

    ``train_upsamplers=None`` means the CLI option was omitted, so resume restores
    the saved policy. To inspect a legacy optimizer, pass counts of parameter
    *tensors* (not their numel), after freezing CLIP and retaining legacy log_c
    entries, as ``{'frozen': n, 'unfrozen': m}``.
    """
    if train_upsamplers is not None:
        _boolean(train_upsamplers, "train_upsamplers")
    _boolean(initial_weights, "initial_weights")
    if checkpoint is None:
        fine_tuning = initial_weights
        frozen = fine_tuning and train_upsamplers is not True
        return {"version": 1, "fine_tuning": fine_tuning, "freeze_upsamplers": frozen}
    if initial_weights:
        _fail("--resume and --weights cannot be combined.")

    layout_frozen = _optimizer_freeze_state(checkpoint, optimizer_param_counts)
    state = checkpoint.get("training_state")
    if state is not None:
        if not isinstance(state, Mapping) or state.get("version") != 1:
            _fail("Unsupported checkpoint training_state version.")
        fine_tuning = _boolean(state.get("fine_tuning"), "saved fine_tuning")
        frozen = _boolean(state.get("freeze_upsamplers"), "saved freeze_upsamplers")
        if frozen and not fine_tuning:
            _fail("Checkpoint freezes pretrained upsamplers but is marked as training from scratch.")
        if layout_frozen is not None and frozen != layout_frozen:
            _fail("Saved freezing policy disagrees with the checkpoint optimizer layout.")
    else:
        saved_args = checkpoint.get("args")
        saved_weights = _argument(saved_args, "weights")
        saved_resume = _argument(saved_args, "resume")
        saved_train_upsamplers = _argument(saved_args, "train_upsamplers", False)
        if saved_train_upsamplers is None:
            saved_train_upsamplers = False
        _boolean(saved_train_upsamplers, "saved train_upsamplers")

        # args.weights is lost after the first legacy resume. A smaller optimizer
        # group is still positive evidence that this was frozen fine-tuning.
        if saved_weights:
            fine_tuning = True
        elif saved_weights == "" and not saved_resume:
            fine_tuning = False
        else:
            fine_tuning = None
        frozen = layout_frozen
        if fine_tuning is not None:
            expected_frozen = fine_tuning and not saved_train_upsamplers
            if frozen is not None and frozen != expected_frozen:
                _fail("Legacy checkpoint arguments disagree with its optimizer layout.")
            frozen = expected_frozen
        elif frozen is True and not saved_train_upsamplers:
            fine_tuning = True
        else:
            _fail("Legacy checkpoint lost its fine-tuning history; the original mode cannot be recovered unambiguously.")

    if train_upsamplers is not None:
        requested_frozen = fine_tuning and not train_upsamplers
        if requested_frozen != frozen:
            _fail("--train_upsamplers conflicts with the checkpoint freezing policy.")
    return {"version": 1, "fine_tuning": fine_tuning, "freeze_upsamplers": frozen}


def make_schedule_plan(steps_per_epoch, total_epochs, warmup=True,
                       warmup_epochs=1, warmup_factor=1e-3, power=0.9,
                       batch_size=None, samples_per_epoch=None):
    """Create the complete serializable input to the polynomial LR function."""
    steps_per_epoch = _integer(steps_per_epoch, "steps_per_epoch", 1)
    total_epochs = _integer(total_epochs, "total_epochs", 1)
    warmup = _boolean(warmup, "warmup")
    warmup_epochs = _integer(warmup_epochs, "warmup_epochs", 0) if warmup else 0
    if warmup_epochs > total_epochs:
        _fail("warmup_epochs cannot exceed total_epochs.")
    warmup_factor = _real(warmup_factor, "warmup_factor", 0.0, 1.0)
    power = _real(power, "power", 0.0)
    if power == 0:
        _fail("power must be positive.")
    if batch_size is not None:
        batch_size = _integer(batch_size, "batch_size", 1)
    if samples_per_epoch is not None:
        samples_per_epoch = _integer(samples_per_epoch, "samples_per_epoch")
    return {
        "version": 1,
        "steps_per_epoch": steps_per_epoch,
        "total_epochs": total_epochs,
        "warmup": warmup,
        "warmup_epochs": warmup_epochs,
        "warmup_factor": warmup_factor,
        "power": power,
        "batch_size": batch_size,
        "samples_per_epoch": samples_per_epoch,
    }


def _normalize_schedule_plan(plan):
    if not isinstance(plan, Mapping) or plan.get("version") != 1:
        _fail("Unsupported checkpoint scheduler_plan version.")
    required = ("steps_per_epoch", "total_epochs", "warmup", "warmup_epochs",
                "warmup_factor", "power")
    missing = [name for name in required if name not in plan]
    if missing:
        _fail("Checkpoint scheduler_plan is missing {}.".format(", ".join(missing)))
    return make_schedule_plan(**{key: plan[key] for key in required},
                              batch_size=plan.get("batch_size"),
                              samples_per_epoch=plan.get("samples_per_epoch"))


def _completed_steps(checkpoint):
    epoch = checkpoint.get("epoch")
    scheduler = checkpoint.get("lr_scheduler")
    if epoch is None or not isinstance(scheduler, Mapping) or "last_epoch" not in scheduler:
        return None, None
    completed_epochs = _integer(epoch, "checkpoint epoch", 0) + 1
    completed_steps = _integer(scheduler["last_epoch"], "scheduler last_epoch", 0)
    if completed_steps == 0 or completed_steps % completed_epochs:
        _fail("Checkpoint scheduler progress is inconsistent with an epoch-boundary checkpoint.")
    return completed_epochs, completed_steps


def _legacy_schedule_plan(checkpoint):
    saved_args = checkpoint.get("args")
    epochs = _argument(saved_args, "epochs")
    if epochs is None:
        _fail("Legacy checkpoint has no original total epoch count.")
    batch_size = _argument(saved_args, "batch_size")
    samples = _argument(saved_args, "samples_per_epoch")
    argument_steps = None
    if batch_size is not None:
        batch_size = _integer(batch_size, "saved batch_size", 1)
    if samples is not None:
        samples = _integer(samples, "saved samples_per_epoch")
        if samples > 0 and batch_size is not None:
            argument_steps = (samples + batch_size - 1) // batch_size
    completed_epochs, completed_steps = _completed_steps(checkpoint)
    observed_steps = None if completed_epochs is None else completed_steps // completed_epochs
    if (argument_steps is not None and observed_steps is not None
            and argument_steps != observed_steps):
        _fail("Legacy dataset arguments disagree with the saved scheduler progress.")
    steps = argument_steps if argument_steps is not None else observed_steps
    if steps is None:
        _fail("Legacy checkpoint has insufficient information to recover steps_per_epoch.")
    # The legacy trainer used these fixed warmup values in create_lr_scheduler.
    return make_schedule_plan(
        steps, epochs, warmup=_argument(saved_args, "warmup", True),
        warmup_epochs=_argument(saved_args, "warmup_epochs", 1),
        warmup_factor=_argument(saved_args, "warmup_factor", 1e-3),
        power=_argument(saved_args, "lr_power", 0.9),
        batch_size=batch_size, samples_per_epoch=samples,
    )


def resolve_schedule_plan(checkpoint, requested_steps_per_epoch=None,
                          requested_epochs=None, requested_batch_size=None,
                          requested_samples_per_epoch=None):
    """Recover the original schedule and reject changes to the experiment.

    Call once before building DataLoaders (omit requested_steps_per_epoch) to
    restore omitted CLI values, and again with len(train_loader) to verify the
    actual number of optimizer steps. Omitted CLI options must remain ``None``;
    argparse defaults must not silently replace saved settings.
    """
    saved_plan = checkpoint.get("scheduler_plan")
    plan = (_legacy_schedule_plan(checkpoint) if saved_plan is None
            else _normalize_schedule_plan(saved_plan))
    requested = {
        "steps_per_epoch": requested_steps_per_epoch,
        "total_epochs": requested_epochs,
        "batch_size": requested_batch_size,
        "samples_per_epoch": requested_samples_per_epoch,
    }
    for key, value in requested.items():
        if value is None:
            continue
        value = _integer(value, "requested " + key,
                         None if key == "samples_per_epoch" else 1)
        if plan[key] is not None and value != plan[key]:
            _fail("--resume requires {}={} from the checkpoint, got {}."
                  .format(key, plan[key], value))
        if plan[key] is None:
            plan[key] = value
    completed_epochs, completed_steps = _completed_steps(checkpoint)
    if completed_epochs is not None:
        if completed_epochs > plan["total_epochs"]:
            _fail("Checkpoint progress exceeds its original schedule horizon.")
        if completed_steps != completed_epochs * plan["steps_per_epoch"]:
            _fail("Checkpoint scheduler progress disagrees with its saved steps_per_epoch.")
    return plan


def polynomial_lr_factor(step, plan):
    """Return a real, nonnegative factor, including at/after the final step.

    Validate/construct the plan once before passing this function to LambdaLR;
    resume compatibility is checked separately, never hidden by this clamp.
    """
    step = _integer(step, "scheduler step", 0)
    total_steps = plan["steps_per_epoch"] * plan["total_epochs"]
    if step >= total_steps:
        return 0.0
    warmup_steps = plan["warmup_epochs"] * plan["steps_per_epoch"] if plan["warmup"] else 0
    if warmup_steps and step <= warmup_steps:
        alpha = float(step) / warmup_steps
        return plan["warmup_factor"] * (1 - alpha) + alpha
    progress = float(step - warmup_steps) / (total_steps - warmup_steps)
    return max(0.0, min(1.0, 1.0 - progress)) ** plan["power"]
