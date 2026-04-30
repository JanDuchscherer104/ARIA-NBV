#!/usr/bin/env bash
set -euo pipefail

cmd="${1:-summary}"
part="${2:-}"

echo "warning: run Slurm status commands as one-shot checks; do not poll in tight loops" >&2

case "$cmd" in
  summary)
    sinfo -o "%20P %8a %12l %6D %8t %30G %N"
    ;;
  gpu)
    sinfo -o "%24P %8t %6D %30G %N" | grep -E "gpu|PARTITION|lrz-|mcml-" || true
    ;;
  cpu)
    sinfo -p lrz-cpu -o "%20P %6D %8t %N"
    ;;
  partition)
    if [[ -z "$part" ]]; then
      echo "usage: $0 partition <partition>" >&2
      exit 2
    fi
    sinfo -p "$part" -o "%20P %8a %12l %6D %8t %30G %N"
    ;;
  nodes)
    if [[ -n "$part" ]]; then
      sinfo -N -p "$part" -o "%24N %24P %10T %8c %10m %30G"
    else
      sinfo -N -o "%24N %24P %10T %8c %10m %30G"
    fi
    ;;
  mine)
    squeue -u "${USER:-}" -o "%.18i %.20P %.30j %.8T %.10M %.6D %R"
    ;;
  jobs)
    if [[ -z "$part" ]]; then
      echo "usage: $0 jobs <partition>" >&2
      exit 2
    fi
    squeue -p "$part" -o "%.18i %.20P %.12u %.30j %.8T %.10M %.6D %R"
    ;;
  accounting)
    sacct -u "${USER:-}" --format=JobID,JobName,Partition,State,Elapsed,AllocGRES,ExitCode
    ;;
  *)
    cat >&2 <<USAGE
usage: $0 [summary|gpu|cpu|partition PART|nodes [PART]|mine|jobs PART|accounting]
Do not run this in tight loops; Slurm commands are expensive on shared systems.
USAGE
    exit 2
    ;;
esac
