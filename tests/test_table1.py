"""Protocol regression checks: run python -m unittest discover -s tests."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = load("core", ROOT/"scripts/_table1_core.py")
runner = load("runner", ROOT/"scripts/reproduce_table1.py")


class Table1Tests(unittest.TestCase):
    def test_release_uses_fixed_partition_manifests(self):
        config = json.loads((ROOT/"configs/table1.json").read_text())
        runs = [run for setting in config["benchmarks"].values() for run in setting["runs"]]
        self.assertEqual(len(runs), 27)
        for run in runs:
            self.assertNotIn("split_seed", run)
            path = ROOT/run["partition"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), run["partition_sha256"])
            fixed = json.loads(path.read_text())
            self.assertEqual(set(fixed), {"train", "val", "test"})
            self.assertFalse(set(fixed["train"]) & set(fixed["val"]))
            self.assertFalse(set(fixed["train"]) & set(fixed["test"]))
            self.assertFalse(set(fixed["val"]) & set(fixed["test"]))

    def test_taskmax_uses_evaluation_task_max_and_excludes_future(self):
        predictions = [
            dict(task_id="a", length=4, label=1, scores=np.array([.1,.9,.99,.99])),
            dict(task_id="a", length=8, label=0, scores=np.array([.7,.7,1,1,1,1,1,1])),
        ]
        auc, values, cutoffs = core.score(predictions)
        self.assertEqual(cutoffs, [2,2])
        self.assertEqual(values, [.9,.7])
        self.assertEqual(auc, 100.)

    def test_one_class_fails_instead_of_reporting_chance(self):
        with self.assertRaises(ValueError):
            core.score([dict(task_id="a", length=4, label=1, scores=np.ones(4))])

    def test_timestamp_does_not_depend_on_length(self):
        episodes = [dict(raw=np.zeros((length,2),dtype=np.float32), length=length,
                         label=0, task_id="a") for length in [2,5]]
        dataset = core.CachedDataset(episodes, {"a":np.zeros(1024,dtype=np.float32)},
                                    np.zeros(2),np.ones(2),5,"pi0_libero")
        np.testing.assert_array_equal(dataset[0][0][:2,-1], dataset[1][0][:2,-1])
        self.assertAlmostEqual(float(dataset[0][0][1,-1]), .01)
        np.testing.assert_array_equal(dataset[0][0][2:], 0)

    def test_checksum_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/"input"
            p.write_text("original")
            expected = runner.digest(p)
            runner.verify_files({str(p):expected})
            p.write_text("modified")
            with self.assertRaises(ValueError):
                runner.verify_files({str(p):expected})


if __name__ == "__main__":
    unittest.main()
