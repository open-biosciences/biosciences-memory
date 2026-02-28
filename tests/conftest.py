"""Shared pytest fixtures for biosciences-memory tests."""

import pytest

from biosciences_memory.config.schema import GraphitiConfig


@pytest.fixture
def default_config():
    """Create a default GraphitiConfig for testing."""
    return GraphitiConfig()
