"""Pytest configuration for vamp-interface."""
import sys
from pathlib import Path

# Add src to sys.path so pytest can find modules
src = Path(__file__).parent / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))
