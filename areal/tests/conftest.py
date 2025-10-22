"""Global pytest configuration and fixtures for areal tests."""

import sys
from unittest.mock import MagicMock

# Mock uvloop for platforms where it's not available (e.g., Windows)
# This must happen BEFORE any imports of areal.api.cli_args
if "uvloop" not in sys.modules:
    mock_uvloop = MagicMock()
    mock_uvloop.install = MagicMock()
    sys.modules["uvloop"] = mock_uvloop
