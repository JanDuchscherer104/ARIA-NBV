#import "symbols.typ": symb

/// SE(3) transform from frame `B` to frame `A` (i.e., `A <- B`).
///
/// Note: Using a function avoids needing whitespace when applying scripts to
/// interpolated symbols (e.g., `$#(symb.vin.T)_A$` would require a space).
#let T(A, B) = $#symb.vin.T^(#A)_(#B)$
