"""Unit tests for the Console utility."""

from oracle_rri.utils import Console, Verbosity


def test_verbosity_is_global() -> None:
    prev_verbosity = Console().verbosity
    prev_debug = Console().is_debug
    try:
        c1 = Console.with_prefix("one")
        c2 = Console.with_prefix("two")

        c1.set_verbosity(Verbosity.QUIET)
        assert c2.verbosity == Verbosity.QUIET

        c2.set_verbosity(Verbosity.VERBOSE)
        assert c1.verbosity == Verbosity.VERBOSE
    finally:
        Console().set_verbosity(prev_verbosity)
        Console().set_debug(prev_debug)


def test_debug_is_global_and_enables_max_verbosity() -> None:
    prev_verbosity = Console().verbosity
    prev_debug = Console().is_debug
    try:
        c1 = Console.with_prefix("one")
        c2 = Console.with_prefix("two")

        c1.set_debug(True)
        assert c2.is_debug is True
        assert c2.verbosity == Verbosity.VERBOSE

        c2.set_debug(False)
        assert c1.is_debug is False
    finally:
        Console().set_verbosity(prev_verbosity)
        Console().set_debug(prev_debug)
