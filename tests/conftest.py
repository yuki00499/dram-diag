import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
for module_name in list(sys.modules):
    if module_name == "dram_diag" or module_name.startswith("dram_diag."):
        del sys.modules[module_name]


_RUN_TEMP_ROOT: Path | None = None


def pytest_configure(config):
    """Give each pytest process its own workspace-local temporary root.

    A fixed base directory can be left with incompatible Windows ACLs when the
    suite is run by different shells, IDEs, or sandboxed agents.  A unique path
    avoids both that ownership collision and parallel-run interference.
    """
    global _RUN_TEMP_ROOT
    _RUN_TEMP_ROOT = Path(config.rootpath) / f".pytest-tmp-{uuid.uuid4().hex}"
    config.option.basetemp = str(_RUN_TEMP_ROOT)


def pytest_unconfigure(config):
    if _RUN_TEMP_ROOT is not None:
        shutil.rmtree(_RUN_TEMP_ROOT, ignore_errors=True)
