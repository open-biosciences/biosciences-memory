"""Unit tests for formatting utilities."""

from unittest.mock import MagicMock

import pytest


@pytest.mark.unit
class TestFormatting:
    def test_format_node_excludes_embedding(self):
        mock_node = MagicMock()
        mock_node.model_dump.return_value = {
            "uuid": "123",
            "name": "BRCA1",
            "attributes": {"name_embedding": [0.1, 0.2]},
        }

        from biosciences_memory.utils.formatting import format_node_result

        result = format_node_result(mock_node)
        assert "name_embedding" not in result.get("attributes", {})

    def test_format_fact_excludes_embedding(self):
        mock_edge = MagicMock()
        mock_edge.model_dump.return_value = {
            "uuid": "456",
            "fact": "BRCA1 interacts with TP53",
            "attributes": {"fact_embedding": [0.3, 0.4]},
        }

        from biosciences_memory.utils.formatting import format_fact_result

        result = format_fact_result(mock_edge)
        assert "fact_embedding" not in result.get("attributes", {})
