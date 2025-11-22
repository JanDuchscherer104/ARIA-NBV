"""Unit tests for the Console utility."""

from oracle_rri.utils import Console


def test_verbose_is_global() -> None:
    prev_verbose = Console().verbose
    try:
        c1 = Console.with_prefix("one")
        c2 = Console.with_prefix("two")

        c1.set_verbose(False)
        assert c2.verbose is False

        c2.set_verbose(True)
        assert c1.verbose is True
    finally:
        Console().set_verbose(prev_verbose)


def test_debug_is_global_and_enables_verbose() -> None:
    prev_verbose = Console().verbose
    prev_debug = Console().is_debug
    try:
        c1 = Console.with_prefix("one")
        c2 = Console.with_prefix("two")

        c1.set_debug(True)
        assert c2.is_debug is True
        assert c2.verbose is True

        c2.set_debug(False)
        assert c1.is_debug is False
    finally:
        Console().set_verbose(prev_verbose)
        Console().set_debug(prev_debug)
