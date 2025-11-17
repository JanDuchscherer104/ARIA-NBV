from typing import Any

import torch
from rich.text import Text
from rich.tree import Tree
from torch import Tensor

from .console import Console


def rich_summary(
    *,
    tree_dict: dict[str, Any],
    path_map: dict[tuple[str, ...], str],
    with_shape: bool = False,
    show_only_sample: list[str] | None = None,
    root_label: str = "",
    is_print: bool = True,
) -> Tree:
    """Build and return a rich Tree from a flattened sample dict.

    - One line per entry; tensors show shape/dtype, optional stats line when
      ``with_shape`` is True (single-element tensors show value).
    - Lists show length/element type (+ first/last for primitive elements).
    - Dicts are traversed recursively; keys listed in ``show_only_sample`` are
      truncated to first/last items.
    - The original flat key is appended to the node label when available.
    """

    def tensor_desc(t: Tensor) -> str:
        return f"{{shape: {tuple(t.shape)}, dtype: {t.dtype}}}"

    def tensor_stats(t: Tensor) -> str:
        if t.numel() == 1:
            return f"{{value: {float(t.item()):.4g}}}"
        if t.numel() == 0 or not t.dtype.is_floating_point:
            return ""
        return f"{{min: {float(t.min()):.4g}, max: {float(t.max()):.4g}, mean: {float(t.mean()):.4g}}}"

    def list_desc(lst: list[Any]) -> str:
        elem_type = type(lst[0]).__name__ if lst else "unknown"
        parts = [f"len: {len(lst)}", f"elem_type: {elem_type}"]
        if lst and not isinstance(lst[0], (dict, list, tuple, torch.Tensor)):
            parts.append(f"first: {lst[0]}")
            parts.append(f"last: {lst[-1]}")
        return "{" + ", ".join(parts) + "}"

    sample_only = set(show_only_sample or [])

    def render(node: Tree, key: str | None, val: Any, path: tuple[str, ...]) -> None:
        lookup_path = path[1:] if path and path[0] == "data" else path
        flat_note = f" [flat: {path_map.get(lookup_path)}]" if lookup_path in path_map else ""
        label = f"{key} <{type(val).__name__}>{flat_note}" if key is not None else None
        current = node if label is None else node.add(Text(label, style="config.field"))

        if isinstance(val, torch.Tensor):
            current.add(Text(tensor_desc(val), style="config.value"))
            if with_shape:
                stats = tensor_stats(val)
                if stats:
                    current.add(Text(stats, style="config.value"))
            return

        if isinstance(val, dict):
            if not val:
                current.add(Text("{}", style="config.value"))
                return
            items = list(val.items())
            if key in sample_only and len(items) > 2:
                items = [items[0], items[-1]]
            for k, v in items:
                render(current, k, v, path + (k,))
            return

        if isinstance(val, (list, tuple)):
            lst = list(val)
            current.add(Text(list_desc(lst), style="config.value"))
            return

        current.add(Text(str(val), style="config.value"))

    root = Tree(Text(root_label, style="config.name"))

    for k, v in tree_dict.items():
        render(root, k, v, (k,))

    if is_print:
        Console().print(root, soft_wrap=False, highlight=True, markup=True, emoji=False)
    return root


def build_nested(
    flat_sample: dict[str, Any], show_semidense: bool = True, show_gt: bool = True
) -> tuple[dict[str, Any], dict[tuple[str, ...], str]]:
    nested: dict[str, Any] = {}
    path_to_flat: dict[tuple[str, ...], str] = {}
    for k, v in flat_sample.items():
        if not show_semidense and k.startswith("msdpd#"):
            continue
        if not show_gt and k.startswith("gt_data"):
            continue

        parts: list[str] = []
        if "#" in k:
            a, rest = k.split("#", 1)
            parts.append(a)
            if "+" in rest:
                b, c = rest.split("+", 1)
                parts.extend([b, c])
            else:
                parts.append(rest)
        else:
            parts.append(k)

        cursor = nested
        for p in parts[:-1]:
            cursor = cursor.setdefault(p, {})
        cursor[parts[-1]] = v
        path_to_flat[tuple(parts)] = k

    return nested, path_to_flat
