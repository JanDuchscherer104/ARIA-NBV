#import "vin_v3_style.typ": diagram_base, input_node, data_node, module_node, decision_node, merge_node, output_node, edge

#diagram_base(
  input_node((1, 0.0), [pose_enc], name: <pose>, width: 28mm),
  input_node((1, 1.4), [global_feat\ (final)], name: <global>, width: 30mm),
  merge_node((2, 0.7), $||$, tint: green, name: <concat>),
  module_node((3, 0.7), [MLP\ (GELU + Dropout)], tint: blue, name: <mlp>, width: 32mm),
  module_node((4, 0.7), [CORAL head], tint: orange, name: <coral>, width: 28mm),
  output_node((5, 0.7), [logits, prob,\ expected RRI], tint: blue, name: <outputs>, width: 34mm),

  input_node((1, 3.0), [voxel_valid_frac], name: <voxel_valid>, width: 30mm),
  input_node((1, 4.2), [semidense_vis_frac], name: <semi_valid>, width: 30mm),
  decision_node((2, 3.6), [valid\ mask], tint: orange, name: <valid_mask>),
  data_node((3, 3.6), [candidate_valid], tint: blue, name: <cand_valid>, width: 30mm),

  edge(<pose>, <concat>, "-|>"),
  edge(<global>, <concat>, "-|>"),
  edge(<concat>, <mlp>, "-|>"),
  edge(<mlp>, <coral>, "-|>"),
  edge(<coral>, <outputs>, "-|>"),

  edge(<voxel_valid>, <valid_mask>, "-|>"),
  edge(<semi_valid>, <valid_mask>, "-|>"),
  edge(<valid_mask>, <cand_valid>, "-|>"),
)
