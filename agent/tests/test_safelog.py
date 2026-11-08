import copy
import io
import logging
import pickle

import pytest

from callback_agent.safelog import PiiExtraFilter, install_pii_filter, log_failure

logger = logging.getLogger("callback_agent.test_safelog")


def raise_it(exc: Exception) -> None:
    raise exc


def failure(exc: Exception) -> Exception:
    try:
        raise_it(exc)
    except Exception as caught:
        return caught
    raise AssertionError("unreachable")


def test_the_type_and_the_place_are_logged_but_not_the_message(caplog):
    with caplog.at_level(logging.ERROR, logger=logger.name):
        log_failure(logger, "forward a turn", failure(ValueError("my code is 482913")))
    [record] = caplog.records
    assert record.levelno == logging.ERROR and record.name == logger.name
    assert record.getMessage().startswith(
        "could not forward a turn: ValueError at test_safelog.py:"
    )
    assert "482913" not in caplog.text and "my code" not in caplog.text


def test_the_place_is_where_it_was_raised_not_where_it_was_caught(caplog):
    with caplog.at_level(logging.ERROR, logger=logger.name):
        log_failure(logger, "x", failure(RuntimeError("boom")))
    raised_at = raise_it.__code__.co_firstlineno + 1
    assert f"test_safelog.py:{raised_at}" in caplog.text


@pytest.mark.parametrize("exc", [KeyError("482913"), TypeError("482913")])
def test_every_exception_is_treated_alike(exc, caplog):
    with caplog.at_level(logging.ERROR, logger=logger.name):
        log_failure(logger, "x", failure(exc))
    assert "482913" not in caplog.text


def test_an_exception_that_was_never_raised_has_no_place_to_name(caplog):
    with caplog.at_level(logging.ERROR, logger=logger.name):
        log_failure(logger, "x", ValueError("482913"))
    assert "ValueError at unknown" in caplog.text and "482913" not in caplog.text


# --- LiveKit's lk.pii extras --------------------------------------------------------------------
#
# LiveKit logs personal data as extra fields named lk.pii.*: a caller's transcript on the warning
# for a dropped turn, the arguments of a tool call. Its formatters print every extra, and a job
# process pickles whole records to the main process. These tests use real records and LiveKit's own
# formatters.

CODE = "my code is 482913"


def record_with(**extra) -> logging.LogRecord:
    """A real record, built the way Logger.makeRecord builds one from `extra`."""
    return logger.makeRecord(
        logger.name, logging.WARNING, __file__, 1, "skipping user input", (), None, extra=extra
    )


def test_the_filter_deletes_every_lk_pii_attribute_and_keeps_the_rest():
    record = record_with(
        **{
            "lk.pii.user_input": CODE,
            "lk.pii.arguments": '{"code": "482913"}',
            "lk.pii": "bare",
            "lk.piiish": "prefix only",
            "lk.job": "j1",
            "call_id": "c1",
            "region": "ca-central",
        }
    )
    assert PiiExtraFilter().filter(record) is True
    left = [name for name in vars(record) if name.startswith("lk.")]
    assert left == ["lk.job"]  # only the personal data goes
    assert (record.call_id, record.region) == ("c1", "ca-central")
    assert (record.msg, record.levelname, record.name) == (
        "skipping user input",
        "WARNING",
        logger.name,
    )


@pytest.mark.parametrize("level", [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR])
def test_the_filter_never_drops_a_record(level):
    clean = record_with(call_id="c1")
    dirty = record_with(**{"lk.pii.transcript": CODE})
    dirty.levelno = clean.levelno = level
    assert PiiExtraFilter().filter(clean) is True
    assert PiiExtraFilter().filter(dirty) is True


def test_a_record_without_personal_data_passes_through_unchanged():
    record = record_with(call_id="c1")
    before = dict(vars(record))
    assert PiiExtraFilter().filter(record) is True
    assert vars(record) == before


def livekit_formatters():
    from livekit.agents.cli.log import ColoredFormatter, JsonFormatter

    dev = ColoredFormatter(
        "%(asctime)s - %(esc_levelcolor)s%(levelname)-4s%(esc_reset)s %(name)s - %(message)s "
        "%(esc_gray)s%(extra)s"
    )
    return {"dev": dev, "production": JsonFormatter()}


@pytest.fixture
def private_logger():
    """A logger that propagates nowhere: a test gives it handlers without touching the real ones."""
    name = "test.pii.private"
    lg = logging.getLogger(name)
    lg.propagate = False
    lg.setLevel(logging.DEBUG)
    yield lg
    lg.handlers.clear()
    lg.filters.clear()
    lg.propagate = True


@pytest.mark.parametrize("which", ["dev", "production"])
def test_livekits_own_formatters_print_the_code_without_the_filter_and_not_with_it(
    which, private_logger
):
    out = io.StringIO()
    handler = logging.StreamHandler(out)
    handler.setFormatter(livekit_formatters()[which])
    private_logger.addHandler(handler)

    private_logger.warning(
        "skipping user input", extra={"lk.pii.user_input": CODE, "call_id": "c1"}
    )
    assert "482913" in out.getvalue()  # the leak the filter is for

    out.truncate(0)
    handler.addFilter(PiiExtraFilter())
    private_logger.warning(
        "skipping user input", extra={"lk.pii.user_input": CODE, "call_id": "c1"}
    )
    printed = out.getvalue()
    assert "482913" not in printed and "lk.pii" not in printed
    assert "skipping user input" in printed and "c1" in printed  # the record itself still gets out


def test_a_record_pickled_for_the_main_process_carries_no_personal_data(private_logger):
    # LogQueueHandler pickles a copy of every record in the job process. The filter runs before
    # a handler's emit, so what it pickles is already clean.
    sent: list[bytes] = []

    class Pickling(logging.Handler):
        def emit(self, record):
            sent.append(pickle.dumps(copy.copy(record)))

    handler = Pickling()
    handler.addFilter(PiiExtraFilter())
    private_logger.addHandler(handler)
    private_logger.warning(
        "skipping user input", extra={"lk.pii.user_input": CODE, "call_id": "c1"}
    )
    [blob] = sent
    assert b"482913" not in blob
    assert pickle.loads(blob).call_id == "c1"


# --- installing the filter ----------------------------------------------------------------------


def livekit_loggers() -> list[logging.Logger]:
    return [
        lg
        for name, lg in logging.root.manager.loggerDict.items()
        if isinstance(lg, logging.Logger) and (name == "livekit" or name.startswith("livekit."))
    ]


def pii_filters(owner) -> list[PiiExtraFilter]:
    return [f for f in owner.filters if isinstance(f, PiiExtraFilter)]


@pytest.fixture
def real_logging_restored():
    """install_pii_filter changes the process's real logging: take the filters off afterwards."""
    yield
    owners = [logging.getLogger(), *livekit_loggers(), *logging.getLogger().handlers]
    for owner in owners:
        for f in pii_filters(owner):
            owner.removeFilter(f)


def printing_handler(out: io.StringIO) -> logging.Handler:
    handler = logging.StreamHandler(out)
    handler.setFormatter(livekit_formatters()["production"])
    return handler


def test_installing_puts_one_filter_on_every_handler_of_the_root_logger(real_logging_restored):
    root = logging.getLogger()
    first, second = logging.NullHandler(), logging.NullHandler()
    root.addHandler(first)
    root.addHandler(second)
    try:
        install_pii_filter()
        install_pii_filter()  # it runs in prewarm and again in each job: twice must not stack
        assert len(pii_filters(first)) == 1 and len(pii_filters(second)) == 1
    finally:
        root.removeHandler(first)
        root.removeHandler(second)


def test_a_handler_on_the_root_logger_prints_no_code_after_installing(real_logging_restored):
    # Through a logger that is not LiveKit's, so that only the handler's filter can be at work.
    out, root = io.StringIO(), logging.getLogger()
    handler = printing_handler(out)
    root.addHandler(handler)
    try:
        install_pii_filter()
        other = logging.getLogger("callback_agent.test_pii_root")
        other.warning("skipping user input", extra={"lk.pii.user_input": CODE})
    finally:
        root.removeHandler(handler)
    assert "skipping user input" in out.getvalue() and "482913" not in out.getvalue()


def test_a_handler_added_after_installing_still_prints_no_code(real_logging_restored):
    # LiveKit adds handlers late when its cloud observability is on (traces.py adds the OTLP log
    # handler inside session.start). The livekit loggers scrub before any handler sees the record.
    install_pii_filter()
    out, root = io.StringIO(), logging.getLogger()
    late = printing_handler(out)
    root.addHandler(late)
    try:
        logging.getLogger("livekit.agents").warning(
            "skipping user input", extra={"lk.pii.user_input": CODE}
        )
    finally:
        root.removeHandler(late)
    assert "skipping user input" in out.getvalue() and "482913" not in out.getvalue()


def test_every_livekit_logger_is_covered_and_only_once(real_logging_restored):
    plugin = logging.getLogger("livekit.plugins.test_pii_plugin")
    install_pii_filter()
    install_pii_filter()
    assert len(pii_filters(logging.getLogger("livekit.agents"))) == 1
    assert len(pii_filters(plugin)) == 1


def test_other_loggers_are_left_alone(real_logging_restored):
    ours = logging.getLogger("callback_agent.test_pii_other")
    install_pii_filter()
    assert pii_filters(ours) == []
