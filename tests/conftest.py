"""Shared pytest setup for tests/."""


def pytest_configure(config):
    # Slow tests still run by default. Skip them for a quick check with:
    #     python -m pytest tests -m "not slow"
    config.addinivalue_line(
        "markers", "slow: takes more than ~10 s (simulations, full grids); "
                   "deselect with -m 'not slow'")
