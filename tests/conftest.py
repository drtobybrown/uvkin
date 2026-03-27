"""
conftest.py — make src/ importable for all tests.
"""
import sys
from pathlib import Path

# Add src/ to sys.path so tests can import pipeline modules
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC_DIR))
