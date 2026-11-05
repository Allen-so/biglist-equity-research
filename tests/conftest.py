"""Private report integration cases are opt-in; no market data is distributed."""
import os
import pytest

def pytest_collection_modifyitems(items):
    for item in items:
        private = (item.name.startswith("test_partial_disclosure") or
                   (item.path.name == "test_publication.py" and item.name.startswith("test_publication[")) or
                   (item.path.name == "test_two_lists.py" and item.name.startswith("test_daily[")))
        if private and os.environ.get("BIGLIST_PRIVATE_TESTS") != "1":
            item.add_marker(pytest.mark.skip(reason="Requires private historical Reports; set BIGLIST_PRIVATE_TESTS=1 only with authorised data"))
