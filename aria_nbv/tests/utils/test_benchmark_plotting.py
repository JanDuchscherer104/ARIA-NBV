"""Tests for Python vs Mojo benchmark plotting helpers."""

from __future__ import annotations

from pathlib import Path

from aria_nbv.utils.benchmark_plotting import (
    BenchmarkRecord,
    build_latency_figure,
    build_scaling_figure,
    build_speedup_figure,
    build_throughput_figure,
    load_benchmark_csv,
    summarize_benchmarks,
)


def test_load_and_summarize_benchmark_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "bench.csv"
    csv_path.write_text(
        "\n".join(
            [
                "benchmark,implementation,problem_size,latency_ms,trial",
                "raycast,python,1024,12.0,0",
                "raycast,python,1024,10.0,1",
                "raycast,mojo,1024,4.0,0",
                "raycast,mojo,1024,5.0,1",
                "compact,python,2048,20.0,0",
                "compact,mojo,2048,8.0,0",
            ]
        ),
        encoding="utf-8",
    )

    records = load_benchmark_csv(csv_path)
    summaries = summarize_benchmarks(records)

    if len(records) != 6:
        raise AssertionError(f"Expected 6 records, got {len(records)}.")
    raycast_python = next(item for item in summaries if item.benchmark == "raycast" and item.implementation == "python")
    if abs(raycast_python.latency_median_ms - 11.0) > 1e-6:
        raise AssertionError(f"Expected Python raycast median 11.0 ms, got {raycast_python.latency_median_ms}.")


def test_speedup_figure_contains_expected_ratio(tmp_path: Path) -> None:
    records = load_benchmark_csv_from_rows(
        tmp_path,
        [
            ("raycast", "python", "1024", "12.0"),
            ("raycast", "python", "1024", "10.0"),
            ("raycast", "mojo", "1024", "4.0"),
            ("raycast", "mojo", "1024", "5.0"),
        ],
    )
    summaries = summarize_benchmarks(records)
    fig = build_speedup_figure(summaries)
    if len(fig.data) != 1:
        raise AssertionError(f"Expected one speedup trace, got {len(fig.data)}.")
    speedup = float(fig.data[0].y[0])
    if abs(speedup - (11.0 / 4.5)) > 1e-6:
        raise AssertionError(f"Unexpected speedup value: {speedup}.")


def test_latency_scaling_and_throughput_figures_emit_expected_traces(tmp_path: Path) -> None:
    records = load_benchmark_csv_from_rows(
        tmp_path,
        [
            ("raycast", "python", "1024", "12.0", "120000"),
            ("raycast", "mojo", "1024", "5.0", "320000"),
            ("raycast", "python", "2048", "24.0", "118000"),
            ("raycast", "mojo", "2048", "10.0", "315000"),
            ("compact", "python", "1024", "9.0", "171000"),
            ("compact", "mojo", "1024", "3.0", "448000"),
            ("pose_score", "python", "1024", "18.0", "91000"),
            ("pose_score", "mojo", "1024", "7.0", "229000"),
        ],
    )
    summaries = summarize_benchmarks(records)
    latency_fig = build_latency_figure(summaries)
    scaling_fig = build_scaling_figure(summaries)
    throughput_fig = build_throughput_figure(summaries)

    if len(latency_fig.data) != 2:
        raise AssertionError(f"Expected two latency traces, got {len(latency_fig.data)}.")
    if len(scaling_fig.data) != 6:
        raise AssertionError(f"Expected six scaling traces, got {len(scaling_fig.data)}.")
    if len(throughput_fig.data) != 2:
        raise AssertionError(f"Expected two throughput traces, got {len(throughput_fig.data)}.")

    if scaling_fig.layout.xaxis.title.text not in (None, ""):
        raise AssertionError("Expected only the bottom scaling subplot to carry an x-axis title.")
    if scaling_fig.layout.xaxis2.title.text not in (None, ""):
        raise AssertionError("Expected middle scaling subplot x-axis title to be omitted.")
    if scaling_fig.layout.xaxis3.title.text != "Problem size":
        raise AssertionError("Expected bottom scaling subplot x-axis title to be preserved.")


def load_benchmark_csv_from_rows(
    tmp_path: Path,
    rows: list[tuple[str, str, str, str] | tuple[str, str, str, str, str]],
) -> list[BenchmarkRecord]:
    csv_path = tmp_path / "bench_rows.csv"
    has_throughput = len(rows[0]) == 5

    header = ["benchmark", "implementation", "problem_size", "latency_ms"]
    if has_throughput:
        header.append("throughput_items_per_s")

    csv_path.write_text(
        "\n".join(
            [
                ",".join(header),
                *[",".join(row) for row in rows],
            ]
        ),
        encoding="utf-8",
    )
    return load_benchmark_csv(csv_path)
