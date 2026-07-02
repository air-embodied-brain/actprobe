"""Build data/task_instructions.json from metrics_logs jsonl.

每个 jsonl 第一条 episode 包含 task_description；按 task_id 取一份 → JSON。
"""
import json
import os
import re
from pathlib import Path

ROOT = Path(os.environ["PI0_ROOT"])
SRC  = ROOT / "data" / "metrics_logs"
OUT  = ROOT / "data" / "task_instructions.json"


def main():
    instructions = {}
    for p in sorted(SRC.glob("task_*.jsonl")):
        m = re.match(r"task_(\d+)\.jsonl", p.name)
        if not m:
            continue
        tid = int(m.group(1))
        with open(p) as f:
            ep = json.loads(f.readline())
        instructions[f"task_{tid}"] = ep["task_description"]

    OUT.write_text(json.dumps(instructions, indent=2, ensure_ascii=False))
    print(f"Wrote {len(instructions)} entries → {OUT}")
    for k, v in sorted(instructions.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
