"""
General configuration for tests.
"""
import glasflow


def pytest_sessionstart():
    """Log which nflows backend is being using"""
    print(f"glasflow config: USE_NFLOWS={glasflow.USE_NFLOWS}")
