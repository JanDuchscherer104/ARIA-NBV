#import "vin_v3_style.typ": diagram_base, input_node, data_node, module_node, op_node, edge

#diagram_base(
  input_node((1, 0.0), [PoseTW\ candidate + reference], name: <poses>),
  module_node((2, 0.0), [Pose encoder\ R6D + LFF], tint: blue, name: <pose_enc>, width: 32mm),
  data_node((3, 0.0), [pose_enc\ (B, N, F_pose)], tint: blue, name: <pose_out>, width: 32mm),

  input_node((1, 1.6), [field + pts_world], name: <field_pts>),
  op_node((2, 1.6), [pos_grid\ from pts_world], tint: orange, name: <posgrid>),
  module_node((3, 1.6), [PoseConditioned\ GlobalPool], tint: green, name: <pool>, width: 32mm),
  data_node((4, 1.6), [global_feat], tint: blue, name: <global>),

  input_node((1, 3.2), [counts_norm +\ candidate centers], name: <counts>),
  op_node((2, 3.2), [sample\ voxel field], tint: orange, name: <sample>),
  data_node((3, 3.2), [voxel_valid_frac], tint: blue, name: <valid>, width: 28mm),
  module_node((4, 3.2), [Linear + Sigmoid\ voxel gate], tint: teal, name: <gate>, width: 32mm),
  data_node((5, 3.2), [global_feat\ (gated)], tint: blue, name: <global_gated>, width: 32mm),

  edge(<poses>, <pose_enc>, "-|>"),
  edge(<pose_enc>, <pose_out>, "-|>"),

  edge(<field_pts>, <posgrid>, "-|>"),
  edge(<posgrid>, <pool>, "-|>"),
  edge(<pool>, <global>, "-|>"),
  edge(<pose_out>, <pool>, "--|>"),

  edge(<counts>, <sample>, "-|>"),
  edge(<sample>, <valid>, "-|>"),
  edge(<valid>, <gate>, "-|>"),
  edge(<global>, <gate>, "-|>"),
  edge(<gate>, <global_gated>, "-|>"),
)
