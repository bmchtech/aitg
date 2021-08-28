from pathlib import Path
from single_source import get_version

__version__ = get_version(__name__, Path(__file__).parent.parent)
# print('ver:', __version__, 'path:', Path(__file__).parent.parent)