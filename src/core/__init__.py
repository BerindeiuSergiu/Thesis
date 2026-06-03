"""Legacy core namespace.

The active application architecture lives in ``src.app``, ``src.application``,
``src.models``, ``src.pipeline``, and ``src.viewer``. This package is kept
importable so legacy namespace checks do not fail on missing historical modules.
"""

__all__: list[str] = []
