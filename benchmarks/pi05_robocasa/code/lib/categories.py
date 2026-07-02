"""Task → category mapping for pi0.5 + RoboCasa (24 tasks, single-stage)."""

CATEGORIES = {
    "PnP":    ["PnPCounterToCab", "PnPCabToCounter", "PnPCounterToSink",
               "PnPSinkToCounter", "PnPCounterToMicrowave",
               "PnPMicrowaveToCounter", "PnPCounterToStove", "PnPStoveToCounter"],
    "Turn":   ["TurnOnSinkFaucet", "TurnOffSinkFaucet", "TurnSinkSpout",
               "TurnOnStove", "TurnOffStove", "TurnOnMicrowave", "TurnOffMicrowave"],
    "Open":   ["OpenSingleDoor", "OpenDoubleDoor", "OpenDrawer"],
    "Coffee": ["CoffeeSetupMug", "CoffeeServeMug", "CoffeePressButton"],
    "Close":  ["CloseDoubleDoor", "CloseDrawer", "CloseSingleDoor"],
}

TASK2CAT = {t: c for c, ts in CATEGORIES.items() for t in ts}

# Paper-specific decision: pi0.5 keeps all 24 tasks (no exclusions).
EXCLUDE_TASKS = set()

ALL_TASKS = sorted([t for ts in CATEGORIES.values() for t in ts])
assert len(ALL_TASKS) == 24, f"expected 24 tasks, got {len(ALL_TASKS)}"
