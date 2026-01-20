#import "@preview/booktabs:0.0.4": *
#show: booktabs-default-table-style

= Prototype Training Configuration (Future Work)

This table records a prototype starting configuration for future VIN-style
training runs. The central contribution of this paper is oracle label
computation; learning a next-best-view policy remains future work. We include
these hyperparameters to make planned experiments reproducible. We use AdamW
with a OneCycle learning-rate policy @OneCycleLR-smith2018 and include a
lightweight coverage-weight curriculum to reduce early collapse.

#figure(
  kind: "table",
  supplement: [Table],
  caption: [Prototype training configuration for future VIN-style experiments.],
  table(
    columns: (18em, auto),
    align: (left, left),
    toprule(),
    table.header([Parameter], [Value]),
    midrule(),
    [Number of classes], [15],
    [Head hidden dim], [192],
    [Head layers], [3],
    [Head dropout], [0.05],
    [Field dim], [16],
    [Global pool grid], [6×6×6],
    [Semidense frustum MHCA], [enabled],
    [Semidense obs. count feature], [enabled],
    [Trajectory encoder], [enabled],
    [Point encoder (PointNeXt)], [optional (off by default)],
    [Voxel valid-frac gate], [enabled],
    [Voxel valid-frac feature], [enabled],
    [Optimizer], [AdamW],
    [Learning rate], [3e-4],
    [Weight decay], [1e-3],
    [Scheduler], [OneCycleLR],
    [Max LR], [3e-4],
    [pct_start], [0.15],
    [div_factor], [15],
    [final_div_factor], [5e3],
    [Gradient clip], [3.0],
    [Coverage weighting], [voxel (anneal 0.6→0 over 5 epochs)],
    [Aux loss], [Huber],
    [Aux weight gamma], [0.90],
    [CORAL bias init], [prior logits],
    bottomrule(),
  ),
) <tab:train-config>
