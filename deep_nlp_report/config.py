
from types import SimpleNamespace
from pathlib import Path

from .utils import this_scripts_path


_data_root = (this_scripts_path() / '../data').resolve()
 

args = SimpleNamespace(
  data_root= _data_root,
  definitions_path= _data_root / 'definitions/definitions.tok',
)

