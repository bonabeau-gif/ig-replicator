# -*- coding: utf-8 -*-
"""Agent and command-line layer for the igr package (`igr` console script).

Imports nothing heavy at module level so `igr describe` and `igr --help` stay fast.
See AGENTS.md at the repository root.
"""
import sys

sys.dont_write_bytecode = True   # a run must not write files (bytecode caches) outside its run directory

__version__ = "1.0.0"
