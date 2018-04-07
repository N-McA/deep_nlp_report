
import os
from pathlib import Path

def this_scripts_path():
  return Path(os.path.dirname(os.path.realpath(__file__)))
