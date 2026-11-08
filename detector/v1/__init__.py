"""The agent contract, callback_shared 1.1.0, served under /v1.

Imports callback_shared only, never the internal shared.contract: the two
define different models under the same names, and the /v1 wire is the shared
package's. tests/v1/test_imports.py holds that line.
"""

from __future__ import annotations

from detector.v1.router import build_v1_router
from detector.v1.service import V1Service

__all__ = ["V1Service", "build_v1_router"]
