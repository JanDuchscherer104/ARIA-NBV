# Slide update: rendering + backprojection

- Integrated depth rendering figures (cand_renders_1x3, depth_histograms_3x3) into the main "Candidate Depth Rendering" slide alongside IO + ops and key params.
- Reworked "Backprojection" slide to include NDC mapping + unprojection equations and the backproj+semi figure.
- Removed the separate "Rendering diagnostics" slide to keep figures with IO + theory.
- `make typst-slide SLIDES_FILE=slides_4.typ` succeeds after fixing the NDC unprojection line (avoid `#symb.oracle.depth_q(u, v)` function call).
