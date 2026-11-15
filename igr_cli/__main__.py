# -*- coding: utf-8 -*-
"""`python -m igr_cli ...` is the same as the `igr` console script."""
import sys

from .cli import main

sys.exit(main())
