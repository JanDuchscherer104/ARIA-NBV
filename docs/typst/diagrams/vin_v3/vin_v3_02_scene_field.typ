#import "vin_v3_style.typ": diagram_base, input_node, data_node, module_node, op_node, act_node, edge

#diagram_base(
  input_node((1, 0), [EvlBackboneOutput\ occ_pr, cent_pr\ counts, occ_input\ free_input], name: <backbone>, width: 36mm),
  op_node((2, 0), [derive channels\ counts_norm, unknown\ new_surface_prior], tint: orange, name: <derive>),
  op_node((3, 0), [concat field_in\ selected channels], tint: orange, name: <concat>),
  module_node((4, 0), [1x1x1 Conv3d], tint: blue, name: <conv>),
  act_node((5, 0), [GroupNorm\ + GELU], tint: yellow, name: <act>),
  data_node((6, 0), [field\ (B, C, D, H, W)], tint: blue, name: <field>, width: 32mm),
  data_node((3, 1.2), [field_in\ (B, C_in, D, H, W)], tint: gray, name: <field_in>, width: 32mm),

  edge(<backbone>, <derive>, "-|>"),
  edge(<derive>, <concat>, "-|>"),
  edge(<concat>, <conv>, "-|>"),
  edge(<conv>, <act>, "-|>"),
  edge(<act>, <field>, "-|>"),
  edge(<concat>, <field_in>, "-|>"),
)
