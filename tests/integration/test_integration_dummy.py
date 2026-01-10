"""Placeholder integration test for CI.

Real integration tests that call external LLMs are in other files in this
directory and are marked with @pytest.mark.slow to exclude them from CI.

Run slow tests locally with: pytest -m slow tests/integration/ -v
"""

import pytest


# Placeholder test to ensure integration test directory runs in CI.
@pytest.mark.integration
def test_integration_placeholder():
    """Ensure CI can run integration test directory without failures."""
    assert True
