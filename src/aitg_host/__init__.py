from pathlib import Path
import os
from single_source import get_version

_ver_path = Path(__file__).parent.parent
__version__ = get_version(__name__, _ver_path)

if os.environ.get("DEBUG") == "1":
    print('ver:', __version__, '_ver_path:', _ver_path)