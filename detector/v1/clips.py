"""Where enrollment audio comes from, and why there is none yet.

`/v1` is built around a hidden listener: the detector joins the LiveKit room,
hears the caller's track, and resolves challenges from what it heard. Nothing
on the wire carries audio. There is no `AudioRef` in `callback_shared`, and a
challenge is issued and then *polled* rather than answered with a body, so
the agent has no way to hand over a clip even if it wanted to. That is a
design decision, not an omission.

The listener is S3 stretch work and is not in Round 2 (ruling conflicts 16
and 21, and the open items in section 6). So in real mode every challenge
resolves `failed`/`no_audio`, no enroll prompt can pass, and an enrollment
can never reach the five passes it needs. There is nothing to make a print
from, and this module does not pretend otherwise.

What it does is name the seam, so that the enrollment write path above it is
real code with tests rather than a comment saying "stretch work". Everything
from a clip to a stored print is built and exercised. When the listener
lands, it implements `ClipSource` and gets passed to `V1Service`; nothing
else has to change.

The contract this sets for whoever writes the listener:

* keep the caller's audio for a challenge until the enrollment completes, and
  no longer. An enrollment is at most 21 challenges and completes in a
  minute or two, so this is a short-lived buffer, not a recording store;
* key it by `challenge_id`, which is what the enrollment knows;
* return mono samples and their sample rate, or None when that challenge was
  not heard. None is the normal answer for a challenge that failed;
* never return audio for a challenge in a session that is locked or ended.
  `V1Service` checks both before asking, but a listener holding audio past a
  deny is a liability whatever its caller does.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = ["ClipSource"]


@runtime_checkable
class ClipSource(Protocol):
    """The caller's audio for one resolved challenge.

    Narrow on purpose: one method, taking the one identifier the enrollment
    has, returning the one thing the analysis needs. A test double is three
    lines, which is what makes the path above it testable with no listener,
    no LiveKit room and no audio pipeline.
    """

    def clip_for(self, challenge_id: str) -> tuple[Any, int] | None:
        """Mono samples and their sample rate, or None if nothing was heard.

        `Any` rather than `np.ndarray` because `detector.analysis` accepts a
        numpy array, a torch tensor or any nested sequence of numbers and
        downmixes multi-channel input itself. A listener should not have to
        import numpy to satisfy this.
        """
        ...
