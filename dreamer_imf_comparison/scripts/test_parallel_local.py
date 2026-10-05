"""Numerical and protocol regressions without simulator or cluster access."""

from pathlib import Path
import sys
import unittest

root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root / "dreamer_imf_comparison"), str(root / "imf_dreamer_jax/src")]
suites = []
for relative in ("imf_dreamer_jax/tests", "dreamer_imf_comparison/tests"):
    suites.append(
        unittest.TestLoader().discover(
            str(root / relative), pattern="test_parallel_*.py"
        )
    )
result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(suites))
if not result.wasSuccessful():
    sys.exit(1)
print("PARALLEL_LOCAL_REGRESSIONS_VERIFIED", result.testsRun, flush=True)
