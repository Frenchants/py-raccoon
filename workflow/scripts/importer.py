from pathlib import Path
import sys

# Adjust the path to point from script.py to src/py_raccoon
sys.path.append(str((Path(__file__).parent.parent.parent / 'src').resolve()))