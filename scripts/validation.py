"""Deterministic validation coverage without dependence on filename prefixes."""


def select_validation_samples(task_keys, task_lengths, max_samples=0):
    """Use all pairs by default; explicitly capped runs cover every task."""
    keys = [key for key in task_keys if task_lengths[key] > 0]
    total = sum(task_lengths[key] for key in keys)
    if max_samples <= 0 or max_samples >= total:
        return [(key, index) for key in keys for index in range(task_lengths[key])]
    if max_samples < len(keys):
        raise ValueError("max_val_samples must be 0 (all pairs) or at least the number of validation tasks.")

    # Allocate a balanced budget, redistributing the budget of exhausted tasks.
    counts = dict.fromkeys(keys, 0)
    remaining = int(max_samples)
    while remaining:
        for key in keys:
            if counts[key] < task_lengths[key]:
                counts[key] += 1
                remaining -= 1
                if not remaining:
                    break
    # Sample across each task's complete sorted range, not just its first files.
    return [
        (key, ((2 * index + 1) * task_lengths[key]) // (2 * counts[key]))
        for key in keys for index in range(counts[key])
    ]
