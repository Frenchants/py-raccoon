from pathlib import Path
import sys
import numpy as np

# Adjust the path to point from script.py to src/py_raccoon
sys.path.append(str((Path(__file__).parent.parent.parent / 'src').resolve()))
sys.path.append(str((Path(__file__).parent.parent.parent.parent / 'cycleindex').resolve()))
sys.path.append(str((Path(__file__).parent.parent.parent / 'cycleindex').resolve()))
