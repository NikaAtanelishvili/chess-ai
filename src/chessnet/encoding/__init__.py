"""Board and move encoding from the side-to-move perspective.

Any change to the input planes, the move index or the packed record format must bump
``ENCODING_VERSION``. Datasets and checkpoints record the version they were built with
and refuse to load under a different one.
"""

from typing import Final

ENCODING_VERSION: Final = 1
