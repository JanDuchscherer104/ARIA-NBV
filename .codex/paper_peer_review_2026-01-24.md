# Peer Review: docs/typst/paper/main.typ (2026-01-24)

## Scope
Review covers `docs/typst/paper/main.typ` and all sections included there:
`01-introduction`, `02-related-work`, `03-problem-formulation`, `04-dataset`,
`05-coordinate-conventions`, `05-oracle-rri`, `06-architecture`,
`08a-frustum-pooling`, `07-training-objective`, `07a-binning`,
`07b-training-config`, `08-system-pipeline`, `09-diagnostics`,
`09a-evaluation`, `09b-ablation`, `10-discussion`, `10a-entity-aware`,
`11-conclusion`, and appendices `12c`, `12d`, `12e`, `12`, `12b`.

## Task List
- [x] Read `docs/typst/paper/main.typ` and included section files.
- [x] Check structure, consistency, technical accuracy, and citation coverage.
- [x] Provide section-by-section critique with actionable recommendations.

## Overall Summary
The paper clearly frames the motivation for RRI-driven NBV and provides an
excellent, code-faithful description of the oracle label pipeline and
supporting diagnostics. The document reads more like a system/engineering
report than a results paper, which is fine if positioned explicitly. The
strongest parts are the pipeline detail (appendix) and the clarity around
oracle computation. The weakest parts are (i) ambiguity about what is already
implemented vs. proposed, (ii) inconsistent ordering/numbering of sections, and
(iii) a few technical inconsistencies around CORAL probabilities, camera model
assumptions, and the Chamfer formulation. Addressing those will substantially
improve clarity and defensibility.

## Strengths
- Clear motivation for quality-driven NBV and why coverage proxies fail in
  egocentric indoor scenes.
- Oracle RRI pipeline is described in a reproducible, code-aligned way.
- Diagnostics section and appendix figures are valuable for debugging geometry.
- Architecture sketch is coherent and thoughtfully modularized into ablations.

## Major Issues (Address Before Submission)
1) **Implementation vs. proposal ambiguity**
   - Multiple sections mix implemented features with planned modules, making it
     hard to discern which components exist in code vs. are design sketches.
   - Recommendation: add explicit “Implemented” vs “Planned” tags per module,
     or a short subsection summarizing the current implementation status.

2) **Section ordering/numbering inconsistency**
   - `08a-frustum-pooling` appears *before* `07-*` sections in `main.typ`.
   - Multiple sections share the same numeric prefix (05-coordinate and
     05-oracle-rri). `12b` comes after `12` and `12c/d/e`.
   - Recommendation: reorder includes to match numbering and update filenames
     or titles to avoid confusion in the TOC and citations.

3) **CORAL probability notation mismatch**
   - `07-training-objective` correctly describes CORAL as cumulative
     probabilities, but `09a-evaluation` later treats `p` as *class*
     probabilities without stating the conversion.
   - Recommendation: rename cumulative probabilities (e.g., `c_k`) and define
     class probabilities `pi_k` once, then use `pi_k` consistently in eval.

4) **Camera model and projection clarity**
   - Several sections say “pinhole camera model” for projection, but EFM3D
     uses fisheye intrinsics. You also mix “screen-space” and “NDC” without a
     clear statement of the renderer’s convention.
   - Recommendation: explicitly state that you project with the renderer’s
     camera model for alignment (e.g., PyTorch3D screen-space) and justify any
     approximation if fisheye is replaced by pinhole.

5) **Chamfer formulation vs. mesh sampling**
   - The mesh→point term is described as triangle-to-point distance averaged
     over faces, which implicitly assumes a particular sampling/weighting.
   - Recommendation: clarify whether faces are area-weighted or if you sample
     points on the mesh. If you use PyTorch3D’s point-to-mesh utilities, cite
     the exact formulation or add a brief note.

## Minor Issues / Consistency
- Inconsistent hyphenation: “semi-dense” vs. “semidense” (e.g., `06` and `08a`).
- Repetition of Project Aria ecosystem description across intro and related work.
- Redundant `#import "macros.typ"` in `main.typ` (imported twice).
- “EFM3D/EVL foundation model stack” in intro could be defined once and reused.
- Check that all cited works exist in `docs/references.bib` (e.g.,
  `ProjectAria-ASE-2025`, `ATEK-SurfaceRecon-2025`, `PyTorch3D-Cameras-2025`).
- `08-system-pipeline`: “Direction sampler” row mentions `uniform sphere` and
  `kappa` in the same line; kappa is only meaningful for biased sampling.
- `07a-binning`: stage definition is not specified (frame index? percentage of
  trajectory?). Add a sentence to define `s` precisely.
- `09-diagnostics`: mentions model collapse but no model is trained yet. This
  should be framed as expectations or future diagnostics, not current findings.
- `10-discussion`: includes claims about labeler multiprocessing; ensure they
  are supported by current implementation or mark as future work.

## Section-by-Section Comments

### `01-introduction.typ`
- Strong motivation and positioning. Consider trimming redundant details about
  Project Aria (already covered in related work) to keep intro crisp.
- The contribution list could explicitly state “no learned policy results yet”
  to prevent reviewer expectation mismatch.
- Consider a sentence clarifying the scope: this is a pipeline/diagnostics
  paper with oracle labels rather than a full NBV benchmark.

### `02-related-work.typ`
- Good coverage of VIN-NBV, GenNBV, and EFM3D/EVL.
- Add a short sentence clarifying how your system differs from VIN-NBV beyond
  dataset/trajectory setting (e.g., oracle pipeline + egocentric diagnostics).
- The SceneScript paragraph is long; consider splitting into two shorter
  paragraphs for readability.

### `03-problem-formulation.typ`
- Clear formulation of RRI and candidate selection. Consider adding a note on
  whether `||F||` denotes the number of faces or area-weighted sampling.
- You mention “unprojecting in normalized device coordinates” here; a short
  note that this is a renderer-alignment choice would preempt confusion.

### `04-dataset.typ`
- Explicitly label the ATEK export settings as “internal snapshot” if not part
  of the public release; otherwise provide a citation or config reference.
- Include split information (train/val/test) or clarify that only the GT mesh
  subset is used for oracle labels in this paper.

### `05-coordinate-conventions.typ`
- Good high-level overview. Consider naming the display rotation
  (`rotate_yaw_cw90`) to tie the text to code and diagnostics.
- You mention fisheye camera model but later sections reference pinhole; align.

### `05-oracle-rri.typ`
- Solid description of the oracle pipeline. The numeric example is useful.
- Add a brief sentence clarifying how candidate depth resolution / cropping
  relates to the rendering table in `08-system-pipeline`.

### `06-architecture.typ`
- Clear separation of core vs. optional modules, but you still use assertive
  language (“we compute,” “we use”) that reads like implemented results.
- Suggest a status callout: e.g., “Implemented in VinModelV2: A, B, C; Planned:
  D, E.” This prevents reviewer confusion.
- In the pose encoding section, “Euclidean representation not possible in four
  or fewer dimensions” is correct but could use a concise citation or footnote.

### `08a-frustum-pooling.typ`
- Good description of token structure and masking. Please clarify whether
  `(u,v)` are normalized to `[-1,1]` or pixel-space; this matters for attention.
- Consider noting the maximum token count and how it is chosen.

### `07-training-objective.typ`
- The CORAL explanation is strong. Consider explicitly defining `pi_k` as the
  class probabilities used for evaluation to align with `09a`.
- If you plan to learn bin centers `u_k`, state whether they are initialized
  to quantile means or midpoints and how monotonicity is enforced.

### `07a-binning.typ`
- Clarify how stages are defined (time index, percentage of trajectory, etc.).
- Mention whether stage-wise normalization requires per-scene stats or global
  per-stage stats.

### `07b-training-config.typ`
- Useful to include for reproducibility. Consider a short note stating the
  config is hypothetical/not yet validated to avoid overclaiming.

### `08-system-pipeline.typ`
- Table `@tab:oracle-label-config` is valuable. Consider stating explicitly
  that only the top-`max_candidates_final` are rendered (if pruning happens
  after candidate generation).
- `Direction sampler` row should separate “uniform sphere” and “biased
  Power Spherical (kappa=4)” to avoid contradiction.
- The referenced file `.configs/paper_figures_oracle_labeler.toml` exists;
  consider adding a sentence on how to reproduce the figures (command).

### `09-diagnostics.typ`
- This section reads like results, but the paper does not present trained
  models. Clarify that these diagnostics are *planned* for future training or
  already used in pilot runs if that is the case.

### `09a-evaluation.typ`
- The evaluation definitions are solid. Explicitly mention converting CORAL
  cumulative outputs to class probabilities before computing `Acc@k`.
- Consider clarifying whether Spearman is computed per snippet then averaged
  or globally across all candidates.

### `09b-ablation.typ`
- Good hypothesis-driven ablation list. Add a note about the baseline model
  that all ablations compare against (e.g., EVL + pose + global pooling).

### `10-discussion.typ`
- Good limitations. The computational cost paragraph is strong; consider
  adding a note about approximate surrogates (e.g., subsampling candidates)
  as potential future mitigations.

### `10a-entity-aware.typ`
- Add a citation for ASE/EFM3D providing OBB annotations; otherwise this reads
  as an unsupported claim.
- Consider clarifying whether entity RRI is computed via mesh subset, OBB
  surfaces, or proxy distances.

### `11-conclusion.typ`
- Concise and appropriate. Consider adding one sentence that reiterates that
  results are about oracle labels and diagnostics rather than learned policy
  performance.

### `12c-appendix-oracle-rri-labeler.typ`
- Excellent level of detail. Add a short note on how you handle degenerate
  cases when `bold(z)` aligns with `bold(u)_wup` (roll-stable basis could
  become ill-defined).
- Consider clarifying that collision checks use mesh surfaces (not just
  vertices) if that is the case in code.

### `12d-appendix-vin-v2-details.typ`
- Good clarifications on semi-dense visibility. The definition of
  `semidense_candidate_vis_frac` is clear and helpful.

### `12e-appendix-optuna-analysis.typ`
- Solid description of evidence routines. If these diagnostics are used for
  decisions, consider adding an explicit note on robustness to adaptive trial
  bias (you already mention non-i.i.d.; a short mitigation note would help).

### `12-appendix-gallery.typ` / `12b-appendix-extra.typ`
- Good visual evidence. Consider checking figure order to align with the
  narrative in the diagnostics and oracle sections.

## Not Included in `main.typ`
- `docs/typst/paper/sections/09c-wandb.typ` and
  `docs/typst/paper/sections/example-content.typ` are not included. If they are
  obsolete, consider removing or clearly marking them as archived to avoid
  confusion.

## Suggested Next Steps (Optional)
- Fix section ordering and numbering in `main.typ`.
- Add a short “Implementation Status” paragraph in `06-architecture.typ`.
- Resolve CORAL probability notation in `07` and `09a`.
- Standardize camera model terminology and semi-dense hyphenation.
