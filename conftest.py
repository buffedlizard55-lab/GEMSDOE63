import sys
from pathlib import Path

# Add repo root to python path so pytest works cleanly
sys.path.insert(0, str(Path(__file__).parent))
