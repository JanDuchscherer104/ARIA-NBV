from __future__ import annotations

from enum import Enum

import pytest

from oracle_rri.utils.optuna_optimizable import Optimizable


def test_categorical_single_choice_is_treated_as_fixed() -> None:
    opt = Optimizable.categorical(choices=(True,))

    class Trial:
        def suggest_categorical(self, name: str, choices: list[object]) -> object:  # noqa: ANN401
            raise AssertionError("suggest_categorical should not be called for single-choice categoricals.")

    assert opt.suggest(Trial(), "module_config.vin.use_point_encoder") is True


def test_categorical_multi_choice_delegates_to_trial() -> None:
    opt = Optimizable.categorical(choices=(True, False))

    class Trial:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[object, ...]]] = []

        def suggest_categorical(self, name: str, choices: list[object]) -> object:  # noqa: ANN401
            self.calls.append((name, tuple(choices)))
            return choices[-1]

    trial = Trial()
    assert opt.suggest(trial, "module_config.vin.use_point_encoder") is False
    assert trial.calls == [("module_config.vin.use_point_encoder", (True, False))]


def test_categorical_single_choice_respects_reverse_mapping() -> None:
    class ColorEnum(Enum):
        RED = "red"

    opt = Optimizable.categorical(choices=(ColorEnum.RED,))

    class Trial:
        def suggest_categorical(self, name: str, choices: list[object]) -> object:  # noqa: ANN401
            raise AssertionError("suggest_categorical should not be called for single-choice categoricals.")

    assert opt.suggest(Trial(), "color") is ColorEnum.RED


def test_invalid_single_choice_field_raises_clean_error() -> None:
    opt = Optimizable.categorical(choices=())

    class Trial:
        def suggest_categorical(self, name: str, choices: list[object]) -> object:  # noqa: ANN401
            return None

    with pytest.raises(ValueError, match="at least one choice"):
        opt.suggest(Trial(), "empty")
