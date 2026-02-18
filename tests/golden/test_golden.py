"""Golden tests."""

import sys
from pathlib import Path

import pytest

# Add tests/golden to path
GOLDEN_DIR = Path(__file__).parent
sys.path.append(str(GOLDEN_DIR))

from runner import main as run_golden_suite  # noqa: E402


def test_golden_suite() -> None:
    """Run the Golden Test Suite.

    This ensures that current implementation matches the baselines.
    """
    # Capture SystemExit
    with pytest.raises(SystemExit) as excinfo:
        run_golden_suite(generate_baseline=False)

    assert excinfo.value.code == 0, "Golden Suite Failed"
