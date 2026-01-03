= Toward Entity-Aware NBV

ASE provides object annotations and OBB predictions that enable
entity-conditioned view planning. Instead of optimizing only global
reconstruction quality, we can weight RRI by entity importance and spatial
uncertainty. A simple objective is

#block[
  #align(center)[
    $ "RRI"_"total"(q) = sum_(e in E) w_e "RRI"_e(q) + lambda "RRI"_"scene"(q) $
  ]
]

where $E$ is the set of entities, $w_e$ are user-specified weights, and
$"RRI"_e$ measures improvement on an entity-specific proxy (e.g., distances to
OBB surfaces). This formulation supports task-driven scanning, such as
prioritizing tables, doors, or other objects of interest.

OBB visualizations are provided in the appendix for qualitative context.

Entity-aware RRI is not yet integrated in the current pipeline, but the
architecture already exposes OBB features and attention modules that can
support these objectives.
