#import "@preview/booktabs:0.0.4": *
#show: booktabs-default-table-style

= Training Configuration Snapshot

The current default configuration used for VIN v2 experiments is summarized in
@tab:train-config. These values are tuned to balance convergence stability and
ordinal calibration, and are subject to ongoing ablation.

#figure(
  kind: "table",
  supplement: [Table],
  caption: [Representative training configuration for VIN v2.],
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
    [Optimizer], [AdamW],
    [Learning rate], [3e-4],
    [Weight decay], [1e-3],
    [Scheduler], [OneCycleLR],
    [Max LR], [1e-3],
    [Gradient clip], [8.0],
    [Aux loss], [Huber],
    [Aux weight gamma], [0.90],
    [CORAL bias init], [prior logits],
    bottomrule(),
  ),
) <tab:train-config>

#figure(
  image("/figures/impl/val_loss.png", width: 100%),
  caption: [Example validation loss curve from recent training runs.]
) <fig:val-loss>
