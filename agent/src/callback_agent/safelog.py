"""Logging around a turn. A turn can hold a spoken code, so a failure is logged by what failed."""

import logging
import traceback
from pathlib import Path

PII_PREFIX = "lk.pii"


def log_failure(logger: logging.Logger, action: str, exc: BaseException) -> None:
    """Log that `action` failed, naming the exception's type and where it was raised.

    The message is left out on purpose: it can quote what was being handled (a pydantic
    ValidationError quotes its input), and what was being handled can be a caller's turn.
    """
    frames = traceback.extract_tb(exc.__traceback__)
    where = f"{Path(frames[-1].filename).name}:{frames[-1].lineno}" if frames else "unknown"
    logger.error("could not %s: %s at %s", action, type(exc).__name__, where)


class PiiExtraFilter(logging.Filter):
    """Deletes the lk.pii* attributes of a record, and never drops the record.

    LiveKit logs personal data as extra fields named lk.pii.*: the caller's transcript on the
    warning for a dropped turn, the arguments of a tool call. Its formatters print every extra,
    and a job process pickles whole records, extras included, to the main process. A filter runs
    before either, and it deletes in place, so every handler after it sees a clean record too.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        for name in [n for n in vars(record) if isinstance(n, str) and n.startswith(PII_PREFIX)]:
            delattr(record, name)
        return True


def install_pii_filter() -> None:
    """Put PiiExtraFilter where every LiveKit record passes. Safe to call again.

    Two places, because handlers come and go: on the root logger's handlers, which every record
    from any logger passes, and on the `livekit` loggers, which scrub a record before any handler
    sees it, including one added later (LiveKit's cloud observability adds its own log handler
    inside session.start). Call it once the process's logging is set up: in prewarm, which runs
    after the job process has added its handler, and again at the start of each job.
    """
    root = logging.getLogger()
    livekit = [
        lg
        for name, lg in root.manager.loggerDict.items()
        if isinstance(lg, logging.Logger) and (name == "livekit" or name.startswith("livekit."))
    ]
    for owner in [*root.handlers, *livekit]:
        if not any(isinstance(f, PiiExtraFilter) for f in owner.filters):
            owner.addFilter(PiiExtraFilter())
