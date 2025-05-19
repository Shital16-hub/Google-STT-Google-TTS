"""
Fix circular imports by setting correct import path.
"""
import sys
import os
from pathlib import Path

# Add parent directory to path to ensure imports work properly
current_dir = Path(__file__).parent
root_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))