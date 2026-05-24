import sys
import pathlib

# Inject src/ into sys.path so imports of group_photobooth resolve correctly.
# Must happen in pytest_configure hook to run before test collection.
def pytest_configure(config):
    src_path = str(pathlib.Path(__file__).resolve().parents[2] / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
