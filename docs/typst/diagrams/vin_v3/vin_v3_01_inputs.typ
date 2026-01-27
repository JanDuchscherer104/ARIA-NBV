#import "vin_v3_style.typ": diagram_base, input_node, data_node, module_node, op_node, edge

#diagram_base(
  input_node((1, 3.0), [EFM snippet\ (EfmSnippetView / VinSnippetView)], name: <efm>, width: 34mm),
  module_node((3, 3.0), [EVL backbone\ (frozen, optional)], tint: teal, name: <evl>),
  data_node((5, 3.0), [EvlBackboneOutput\ voxel heads + pts_world], tint: blue, name: <backbone>, width: 34mm),

  input_node((1, 0.0), [Candidate poses\ PoseTW w to cam], name: <cand>, width: 32mm),
  input_node((1, 1.0), [Reference pose\ PoseTW w to rig], name: <ref>, width: 32mm),
  input_node((1, 2.0), [PerspectiveCameras], name: <p3d>, width: 30mm),

  op_node((3, 1.5), [Prepare inputs\ batch + cw90], tint: orange, name: <prep>),
  data_node((5, 1.5), [PreparedInputs\ poses + t_world_voxel], tint: blue, name: <prep_out>, width: 34mm),

  edge(<efm>, <evl>, "-|>"),
  edge(<evl>, <backbone>, "--|>"),
  edge(<efm>, <prep>, "--|>"),
  edge(<cand>, <prep>, "-|>"),
  edge(<ref>, <prep>, "-|>"),
  edge(<p3d>, <prep>, "-|>"),
  edge(<backbone>, <prep>, "-|>"),
  edge(<prep>, <prep_out>, "-|>"),
)
