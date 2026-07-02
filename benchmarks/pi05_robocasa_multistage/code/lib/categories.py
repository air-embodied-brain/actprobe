"""Task → category mapping for pi0.5 + RoboCasa multi-stage (5 long tasks)."""

# 5 multi-stage tasks. Categories = task identity (no grouping).
LONG_TASKS = [
    "ArrangeVegetables",
    "MicrowaveThawing",
    "RestockPantry",
    "PrepareCoffee",
    "PreSoakPan",
]

CATEGORIES = {t: [t] for t in LONG_TASKS}
TASK2CAT = {t: t for t in LONG_TASKS}

EXCLUDE_TASKS = set()
ALL_TASKS = sorted(LONG_TASKS)
assert len(ALL_TASKS) == 5
