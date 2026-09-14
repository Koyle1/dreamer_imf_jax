from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import jax
import numpy as np

from dreamer_imf_compare import flowmpc_actor_study as study
from imf_dreamer_jax import init_rebrac_state


class FlowMPCActorStudyTests(unittest.TestCase):
    def test_self_test(self) -> None:
        study.self_test()

    def test_paper_rebrac_checkpoint_round_trip(self) -> None:
        config_values = study._paper_rebrac_config(6, 2)
        self.assertEqual(config_values["hidden_dim"], 256)
        self.assertEqual(config_values["hidden_layers"], 3)
        self.assertEqual(config_values["batch_size"], 1024)
        self.assertEqual(config_values["policy_frequency"], 2)
        config = study._rebrac_config({"rebrac_config": config_values})
        state = init_rebrac_state(jax.random.key(1), config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pkl"
            study._save_rebrac_checkpoint(path, state, config, {"cell_id": "x"})
            loaded, loaded_config, metadata = study._load_rebrac_checkpoint(path)
        self.assertEqual(loaded_config, config)
        self.assertEqual(metadata, {"cell_id": "x"})
        self.assertEqual(
            study.benchmark._tree_digest(loaded.actor),
            study.benchmark._tree_digest(state.actor),
        )
        self.assertEqual(
            study.benchmark._tree_digest(loaded.critics),
            study.benchmark._tree_digest(state.critics),
        )

    def test_trace_verifier_rejects_changed_action(self) -> None:
        retained = {
            "actions": np.zeros((1, 2, 1), np.float32),
            "lengths": np.asarray([2], np.int32),
        }
        study._assert_trace_close(retained, retained)
        changed = {name: value.copy() for name, value in retained.items()}
        changed["actions"][0, 0, 0] = 0.1
        with self.assertRaisesRegex(ValueError, "actions"):
            study._assert_trace_close(retained, changed)

    def test_tuning_and_evaluation_seed_namespaces_are_disjoint(self) -> None:
        tuning = study.benchmark.derive_seed("flowmpc-tuning", study.TASK)
        evaluation = {
            seed
            for world_seed in (211, 223, 227)
            for actor_seed in (311, 313)
            for seed in study._evaluation_seeds(world_seed, actor_seed)
        }
        self.assertNotIn(tuning, evaluation)
        self.assertEqual(len(evaluation), 30)


if __name__ == "__main__":
    unittest.main()
