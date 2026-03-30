from typing import Any

import torch
from rich.text import Text
from rich.tree import Tree
from torch import Tensor

from .summary import _extract_tensor, _tensor_summary


def summarize(val: Tensor | Any, *, include_stats: bool = False) -> Any:
    """Small helper for succinct repr output."""
    if val is None:
        return None
    if isinstance(val, list):
        return {"len": len(val)}

    tensor = _extract_tensor(val)
    if tensor is not None:
        return _tensor_summary(tensor, include_stats=include_stats)
    return val


def summarize_shape(value: Any) -> str:
    """Return a compact shape/dtype/device string using :func:`summarize`."""
    summary = summarize(value)
    if summary is None:
        return "None"
    if isinstance(summary, dict):
        if "shape" in summary:
            parts = [str(summary["shape"])]
            if "dtype" in summary:
                parts.append(str(summary["dtype"]).replace("torch.", ""))
            if "device" in summary:
                parts.append(str(summary["device"]))
            return " ".join(parts)
        if "len" in summary:
            return f"list(len={summary['len']})"
        return str(summary)
    return type(summary).__name__


def _tensor_desc(tensor: Tensor) -> str:
    return f"{{shape: {tuple(tensor.shape)}, dtype: {tensor.dtype}}}"


def _tensor_stats(tensor: Tensor) -> str:
    if tensor.numel() == 1:
        return f"{{value: {float(tensor.item()):.4g}}}"
    if tensor.numel() == 0 or not tensor.dtype.is_floating_point:
        return ""
    return f"{{min: {float(tensor.min()):.4g}, max: {float(tensor.max()):.4g}, mean: {float(tensor.mean()):.4g}}}"


def _list_desc(items: list[Any]) -> str:
    elem_type = type(items[0]).__name__ if items else "unknown"
    parts = [f"len: {len(items)}", f"elem_type: {elem_type}"]
    if items and not isinstance(items[0], (dict, list, tuple, torch.Tensor)):
        parts.append(f"first: {items[0]}")
        parts.append(f"last: {items[-1]}")
    return "{" + ", ".join(parts) + "}"


def _render_rich_summary(
    node: Tree,
    key: str | None,
    value: Any,
    path: tuple[str, ...],
    *,
    path_map: dict[tuple[str, ...], str],
    show_only_sample: set[str],
    with_shape: bool,
) -> None:
    lookup_path = path[1:] if path and path[0] == "data" else path
    flat_note = f" [flat: {path_map.get(lookup_path)}]" if lookup_path in path_map else ""
    label = f"{key} <{type(value).__name__}>{flat_note}" if key is not None else None
    current = node if label is None else node.add(Text(label, style="config.field"))

    if isinstance(value, torch.Tensor):
        current.add(Text(_tensor_desc(value), style="config.value"))
        if with_shape:
            stats = _tensor_stats(value)
            if stats:
                current.add(Text(stats, style="config.value"))
        return

    if isinstance(value, dict) and _is_tensor_summary(value):
        current.add(Text(_format_tensor_summary(value), style="config.value"))
        return

    if isinstance(value, dict):
        if not value:
            current.add(Text("{}", style="config.value"))
            return
        items = list(value.items())
        if key in show_only_sample and len(items) > 2:
            items = [items[0], items[-1]]
        for k, v in items:
            _render_rich_summary(
                current,
                k,
                v,
                path + (k,),
                path_map=path_map,
                show_only_sample=show_only_sample,
                with_shape=with_shape,
            )
        return

    if isinstance(value, (list, tuple)):
        current.add(Text(_list_desc(list(value)), style="config.value"))
        return

    current.add(Text(str(value), style="config.value"))


def _is_tensor_summary(value: dict[str, Any]) -> bool:
    if "shape" not in value or "dtype" not in value:
        return False
    allowed = {"shape", "dtype", "device", "min", "max", "mean"}
    return set(value.keys()).issubset(allowed)


def _format_tensor_summary(value: dict[str, Any]) -> str:
    shape = value.get("shape")
    dtype = value.get("dtype")
    device = value.get("device")
    parts = [f"Tensor{shape}", str(dtype).replace("torch.", "")]
    if device is not None:
        parts.append(str(device))
    stats = []
    for key in ("min", "max", "mean"):
        if key in value:
            stats.append(f"{key}={value[key]:.4g}")
    if stats:
        parts.append("{" + ", ".join(stats) + "}")
    return " ".join(parts)


def rich_summary(
    tree_dict: dict[str, Any],
    *,
    path_map: dict[tuple[str, ...], str] | None = None,
    with_shape: bool = True,
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

    root = Tree(Text(root_label, style="config.name"))
    summary_data = tree_dict or {}
    resolved_path_map = path_map or {}
    sample_only = set(show_only_sample or [])

    for k, v in summary_data.items():
        _render_rich_summary(
            root,
            k,
            v,
            (k,),
            path_map=resolved_path_map,
            show_only_sample=sample_only,
            with_shape=with_shape,
        )

    if is_print:
        from .console import Console

        Console().print(root, soft_wrap=False, highlight=True, markup=True, emoji=False)
    return root


def capture_tree(tree: Tree) -> str:
    """Render a rich tree into plain text using the project console settings."""
    from .console import Console

    console = Console()
    with console.capture() as capture:
        console.print(
            tree,
            soft_wrap=False,
            highlight=True,
            markup=True,
            emoji=False,
        )
    return capture.get().rstrip()


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
