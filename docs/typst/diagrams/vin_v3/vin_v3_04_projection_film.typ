#import "vin_v3_style.typ": diagram_base, input_node, data_node, module_node, op_node, edge

#diagram_base(
  input_node((4, -1.2), [global_feat\ (gated)], name: <global_in>, width: 30mm),

  input_node((1, 0.0), [voxel points\ (pooled pts_world)], tint: blue, name: <voxel_pts>, width: 32mm),
  op_node((2, 0.0), [project\ voxel centers], tint: orange, name: <vox_proj>),
  data_node((3, 0.0), [voxel proj stats], tint: green, name: <vox_stats>),
  module_node((4, 0.0), [FiLM (voxel)], tint: teal, name: <vox_film>),
  data_node((5, 0.0), [global_feat\ (voxel-mod)], tint: blue, name: <global_vox>, width: 32mm),

  input_node((1, 2.0), [semidense points], tint: blue, name: <semi_pts>),
  op_node((2, 2.0), [project\ semidense], tint: orange, name: <semi_proj>),
  data_node((3, 2.0), [semidense stats], tint: green, name: <semi_stats>),
  module_node((4, 2.0), [FiLM (semidense)], tint: teal, name: <semi_film>, width: 32mm),
  data_node((5, 2.0), [global_feat\ (final)], tint: blue, name: <global_out>, width: 32mm),

  edge(<global_in>, <vox_film>, "-|>"),
  edge(<voxel_pts>, <vox_proj>, "-|>"),
  edge(<vox_proj>, <vox_stats>, "-|>"),
  edge(<vox_stats>, <vox_film>, "-|>"),
  edge(<vox_film>, <global_vox>, "-|>"),

  edge(<global_vox>, <semi_film>, "-|>"),
  edge(<semi_pts>, <semi_proj>, "-|>"),
  edge(<semi_proj>, <semi_stats>, "-|>"),
  edge(<semi_stats>, <semi_film>, "-|>"),
  edge(<semi_film>, <global_out>, "-|>"),
)
