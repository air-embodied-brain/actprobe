"""Task → category mapping for per-category eval (paper Table 2-style)."""

CATEGORIES = {
    "PnP":    ["PnPCounterToCab", "PnPCabToCounter", "PnPCounterToSink",
               "PnPSinkToCounter", "PnPCounterToMicrowave",
               "PnPMicrowaveToCounter", "PnPCounterToStove", "PnPStoveToCounter"],
    "Turn":   ["TurnOnSinkFaucet", "TurnOffSinkFaucet", "TurnSinkSpout",
               "TurnOnStove", "TurnOffStove", "TurnOnMicrowave", "TurnOffMicrowave"],
    "Open":   ["OpenSingleDoor", "OpenDoubleDoor", "OpenDrawer"],
    "Coffee": ["CoffeeSetupMug", "CoffeeServeMug", "CoffeePressButton"],
    "Close":  ["CloseDoubleDoor"],   # CloseDrawer + CloseSingleDoor excluded (zero failures)
}

TASK2CAT = {t: c for c, ts in CATEGORIES.items() for t in ts}

EXCLUDE_TASKS = {"CloseDrawer", "CloseSingleDoor"}

ALL_TASKS = sorted([t for ts in CATEGORIES.values() for t in ts])
assert len(ALL_TASKS) == 22, f"expected 22 tasks, got {len(ALL_TASKS)}"
