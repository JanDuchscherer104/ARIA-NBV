# Lightning Ownership

`aria_nbv.lightning` owns training orchestration for VIN-style scorers:
experiment assembly, datamodules, Lightning modules, trainer factories,
callbacks, optimizers, and package CLIs.

It should compose typed data-handling and VIN contracts rather than redefining
their payload shapes locally.
