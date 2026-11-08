"""Pytest setup shared by every test in this folder."""

# pytest sometimes hangs at exit, after it has printed its summary, while onnxruntime tears down
# its telemetry. The silero plugin brings onnxruntime in (test_attacker_agent.py imports it), and
# no test needs the telemetry, so it is turned off here, before anything can start it. Both
# halves are guarded: without onnxruntime, or with a build that has no such function, there is
# nothing to turn off, and the tests that never touch it must still run.
try:
    import onnxruntime
except ImportError:
    onnxruntime = None

if onnxruntime is not None and hasattr(onnxruntime, "disable_telemetry_events"):
    onnxruntime.disable_telemetry_events()
