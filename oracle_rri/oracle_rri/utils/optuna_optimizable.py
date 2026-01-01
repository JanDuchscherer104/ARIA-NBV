"""Optuna-friendly search space helpers.

This mirrors the utility layer from ``external/doc_classifier`` so oracle_rri
configs can declare Optuna search spaces declaratively via ``optimizable_field``.

The core idea is:

- attach an :class:`Optimizable` instance to a Pydantic ``Field`` via
  ``json_schema_extra={"optimizable": ...}``,
- have an Optuna-aware orchestration layer traverse the config tree and apply
  trial suggestions before constructing runtime objects.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")

if TYPE_CHECKING:
    import optuna


class Optimizable(BaseModel, Generic[T]):
    """Declarative description of an optimisable parameter.

    The class intentionally avoids importing Optuna at runtime so the rest of the
    package can be used without the optional dependency. The ``trial`` argument
    is treated duck-typed (expects ``suggest_float/int/categorical`` methods).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    target: type[Any] | None = Field(
        default=None,
        description="Python type of the parameter (int, float, bool, str, Enum).",
    )
    low: float | int | None = Field(default=None, description="Lower bound for numeric spaces.")
    high: float | int | None = Field(default=None, description="Upper bound for numeric spaces.")
    step: int | None = Field(default=1, description="Step used for discrete integer suggestions.")
    categories: Sequence[Any] | None = Field(
        default=None,
        description="Explicit set of categorical choices (or Enum members).",
    )
    log: bool = Field(default=False, description="Use logarithmic sampling for numeric spaces.")
    name: str | None = Field(default=None, description="Optional override for the Optuna parameter name.")
    default: T | None = Field(default=None, description="Default value used outside Optuna trials.")
    description: str | None = Field(default=None, description="Human readable description of the parameter.")

    @classmethod
    def continuous(
        cls,
        *,
        low: float,
        high: float,
        log: bool = False,
        name: str | None = None,
        default: float | None = None,
        description: str | None = None,
    ) -> "Optimizable[float]":
        return cls(
            target=float,
            low=low,
            high=high,
            log=log,
            name=name,
            default=default,
            description=description,
        )

    @classmethod
    def discrete(
        cls,
        *,
        low: int,
        high: int,
        step: int = 1,
        log: bool = False,
        name: str | None = None,
        default: int | None = None,
        description: str | None = None,
    ) -> "Optimizable[int]":
        return cls(
            target=int,
            low=low,
            high=high,
            step=step or 1,
            log=log,
            name=name,
            default=default,
            description=description,
        )

    @classmethod
    def categorical(
        cls,
        *,
        choices: Sequence[Any],
        name: str | None = None,
        default: Any | None = None,
        description: str | None = None,
    ) -> "Optimizable[Any]":
        return cls(
            categories=tuple(choices),
            name=name,
            default=default,
            description=description,
        )

    def suggest(self, trial: "optuna.Trial", path: str) -> T:  # type: ignore[name-defined]
        """Sample a value from Optuna.

        Args:
            trial: Optuna trial (duck-typed; must implement suggest_* APIs).
            path: Default parameter name derived from the config path.

        Returns:
            Suggested value coerced into the requested target type.
        """
        name = self.name or path
        if self._is_categorical():
            choices = list(self._categorical_choices())
            opt_choices: list[Any] = []
            reverse_map: dict[Any, Any] = {}
            for choice in choices:
                opt_choice, mapped = self._to_optuna_choice(choice)
                opt_choices.append(opt_choice)
                if mapped is not None:
                    reverse_map[opt_choice] = mapped
            value = trial.suggest_categorical(name, opt_choices)
            if value in reverse_map:
                value = reverse_map[value]
            return self._coerce(value)
        if self._is_bool():
            value = trial.suggest_categorical(name, [True, False])
            return self._coerce(value)
        if self._is_int():
            return self._coerce(
                trial.suggest_int(
                    name,
                    int(self._require_low()),
                    int(self._require_high()),
                    step=self.step or 1,
                    log=self.log,
                )
            )
        if self._is_float():
            return self._coerce(
                trial.suggest_float(
                    name,
                    float(self._require_low()),
                    float(self._require_high()),
                    log=self.log,
                )
            )
        raise ValueError(f"Unsupported optimizable configuration for '{path}'.")

    def serialize(self, value: Any) -> Any:
        """Convert a suggested value to a JSON/W&B friendly representation."""
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, (list, tuple)):
            return self._stringify_choice(value)
        return value

    # ------------------------------------------------------------------ helpers
    def _is_bool(self) -> bool:
        return self.target is bool

    def _is_int(self) -> bool:
        return self.target is int

    def _is_float(self) -> bool:
        return self.target is float or (isinstance(self.low, float) or isinstance(self.high, float))

    def _is_categorical(self) -> bool:
        return self.categories is not None or (isinstance(self.target, type) and issubclass(self.target, Enum))

    def _categorical_choices(self) -> Sequence[Any]:
        if self.categories is not None:
            return self.categories
        target = self.target
        if isinstance(target, type) and issubclass(target, Enum):
            return list(target)
        raise ValueError("Categorical optimizables require either categories or an Enum target.")

    def _require_low(self) -> float | int:
        if self.low is None:
            raise ValueError("Optimizable requires 'low'.")
        return self.low

    def _require_high(self) -> float | int:
        if self.high is None:
            raise ValueError("Optimizable requires 'high'.")
        return self.high

    def _coerce(self, value: Any) -> Any:
        target = self.target
        if target is None:
            return value
        if isinstance(target, type) and issubclass(target, Enum):
            if isinstance(value, target):
                return value
            return target(value)
        if target in {int, float, bool, str}:
            return target(value)
        return value

    def _to_optuna_choice(self, choice: Any) -> tuple[Any, Any | None]:
        """Convert a categorical choice into an Optuna-friendly primitive.

        Returns:
            Tuple of (optuna_choice, mapped_value). If mapped_value is not None,
            it is the original choice to restore after sampling.
        """
        if isinstance(choice, Enum):
            return choice.value, choice
        if isinstance(choice, (list, tuple)):
            return self._stringify_choice(choice), choice
        if choice is None or isinstance(choice, (bool, int, float, str)):
            return choice, None
        return str(choice), choice

    def _stringify_choice(self, choice: Sequence[Any]) -> str:
        """Stable string representation for categorical sequences."""
        if all(isinstance(item, str) for item in choice):
            return "+".join(choice)
        if all(isinstance(item, (int, float, bool)) for item in choice):
            return ",".join(str(item) for item in choice)
        return "+".join(str(item) for item in choice)


def optimizable_field(
    *,
    default: T | None = None,
    default_factory: Callable[[], T] | None = None,
    optimizable: Optimizable[T],
    **field_kwargs: Any,
) -> Any:
    """Attach an optimizable definition to a Pydantic Field.

    Exactly one of ``default`` or ``default_factory`` must be provided.
    """
    if (default is None) == (default_factory is None):
        raise ValueError("Provide exactly one of default or default_factory.")
    extras = dict(field_kwargs.pop("json_schema_extra", {}) or {})
    extras["optimizable"] = optimizable
    if default_factory is not None:
        return Field(
            default_factory=default_factory,
            json_schema_extra=extras,
            **field_kwargs,
        )
    return Field(default=default, json_schema_extra=extras, **field_kwargs)


__all__ = [
    "Optimizable",
    "optimizable_field",
]
