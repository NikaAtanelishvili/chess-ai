"""A UCI engine that never answers "go": stands in for a hung engine in timeout tests."""

import sys

for line in sys.stdin:
    words = line.split()
    if not words:
        continue
    if words[0] == "uci":
        print("id name stuck\nuciok", flush=True)
    elif words[0] == "isready":
        print("readyok", flush=True)
    elif words[0] == "quit":
        break
