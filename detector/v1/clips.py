"""Where enrollment audio comes from, and why there is none yet.

`/v1` is built around a hidden listener: the detector joins the LiveKit room,
hears the caller's track, and resolves challenges from what it heard. The
listener is S3 stretch work and is not in Round 2 (ruling conflicts 16 and
21, and the open items in section 6), so in real mode every challenge
resolves `failed`/`no_audio` unless something else supplies the audio.

Something else now can. Contract 1.2.0 added `Turn.audio`, so the agent can
point the detector at the span it just posted a turn for, and an `enroll`
prompt answered with a measurable clip passes. That path needs nothing from
this module.

This module is the other source: audio the detector heard itself, with no
upload. It stays because the listener is still the design, it will have
audio for turns the agent never writes to a bucket, and it is what a passive
voiceprint will eventually read. `V1Service` works with either source, both,
or neither.

The contract this sets for whoever writes the listener:

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
