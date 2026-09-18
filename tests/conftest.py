"""Test configuration.

engine.py and cost_model.py use flat imports (`from cost_model import ...`)
because they run as the container's working directory, so the package dir has
to be on sys.path for tests to import them the same way.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "optimization_engine"))
sys.path.insert(0, str(ROOT / "workload_simulator"))
