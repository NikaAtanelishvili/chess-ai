import os

from hypothesis import settings

# At least 200 examples per property test; no per-example deadline (CI machines vary).
settings.register_profile("default", max_examples=200, deadline=None)
# CI runs are derandomized (fixed seed) so a red build always reproduces.
settings.register_profile("ci", parent=settings.get_profile("default"), derandomize=True)
settings.load_profile("ci" if os.environ.get("CI") else "default")
