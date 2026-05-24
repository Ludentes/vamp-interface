import sys
import pathlib

# Inject scripts/ into sys.path so imports of group_photobooth resolve correctly.
# Must happen in pytest_configure hook to run before test collection.
def pytest_configure(config):
    scripts_path = str(pathlib.Path(__file__).resolve().parents[2] / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
