# Pipeline Ownership

`aria_nbv.pipelines` owns high-level orchestration such as oracle RRI label
generation and cache-producing workflows.

Pipeline code should coordinate existing data, rendering, and metric contracts
instead of creating parallel payload formats or silent compatibility branches.
