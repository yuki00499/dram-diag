import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
for module_name in list(sys.modules):
    if module_name == "dram_diag" or module_name.startswith("dram_diag."):
        del sys.modules[module_name]
