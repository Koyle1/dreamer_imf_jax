import unittest

from dreamer_imf_compare.staged_runner import Schedule


def metrics(latent=0.1, reward=0.1, numerator=2, denominator=1):
    return dict(
        calibration=dict(
            shared_core_model_grad_norm=numerator,
            shared_core_dynamics_grad_norm=denominator,
        ),
        quality_gate=dict(
            normalized_latent_mse_5=latent, normalized_reward_mse_5=reward
        ),
    )


class ScheduleTests(unittest.TestCase):
    def test_boundaries(self):
        s = Schedule()
        self.assertFalse(s.controls(49999)["actor_enabled"])
        self.assertTrue(s.controls(50000)["actor_enabled"])
        self.assertFalse(s.controls(59999)["transition_only"])
        self.assertTrue(s.controls(60000)["transition_only"])
        self.assertFalse(s.controls(100000)["transition_only"])
        self.assertEqual(
            [s.controls(n)["max_gap"] for n in [0, 50000, 100000, 150000]],
            [0.1, 0.25, 0.5, 1],
        )

    def test_gate_has_no_duplicate_credit(self):
        s = Schedule()
        s.observe(100000, metrics())
        s.observe(150000, metrics())
        s.observe(150000, metrics())
        self.assertFalse(s.reliable)
        s.observe(200000, metrics())
        self.assertTrue(s.reliable)
        self.assertEqual(s.controls(200000)["imag_horizon"], 15)
        s.observe(250000, metrics(latent=2))
        self.assertFalse(s.reliable)

    def test_calibration(self):
        s = Schedule()
        s.observe(50000, metrics(numerator=1e10))
        self.assertEqual(s.scale, 5.5)
        s.observe(100000, metrics(denominator=0))
        self.assertEqual(s.scale, 5.5)
        with self.assertRaises(ValueError):
            s.observe(150000, metrics(numerator=float("nan")))


if __name__ == "__main__":
    unittest.main()
