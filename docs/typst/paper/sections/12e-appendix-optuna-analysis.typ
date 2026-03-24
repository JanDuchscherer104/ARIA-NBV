#pagebreak()

= Appendix: Optuna Sweep Evidence & Diagnostics

#import "../../shared/macros.typ": *

#block[
  #smallcaps[Scope] This appendix formalises the evidence routines implemented in
  our Streamlit Optuna Sweep Explorer (`oracle_rri/app/panels/optuna_sweep.py`).
  The intent is to provide fast, decision-oriented signals for which
  hyper-parameters can be fixed and which require further exploration.
]

== Trial filtering

Trials are filtered by Optuna state, finite objective values, and an optional
pandas query; all reported statistics operate on the filtered set.

== Parameter typing

Let $p$ be a parameter series. If at least 80% of its values can be coerced to
numeric and it has more than `max_categories` unique values, we treat it as
*numeric*; otherwise it is *categorical*. Boolean parameters are categorical by
construction.

== Evidence overview (per parameter)

*Categorical parameters.* For categories $A$ and $B$, we compare the best and
runner-up means (direction-aware). Let $bar(y)_A$ and $bar(y)_B$ be their means
with sample sizes $n_A, n_B$ and standard deviations $s_A, s_B$. We report

#block[
  #align(center)[
    $
      "SE" = sqrt((s_A^2)/(n_A) + (s_B^2)/(n_B)), \
      z = (bar(y)_B - bar(y)_A) / "SE"
    $
  ]
]

along with the 95% normal-approximate CI
$[bar(y)_B - bar(y)_A - 1.96 "SE", bar(y)_B - bar(y)_A + 1.96 "SE"]$ and
Cliff's delta (direction-aware) for effect size. A parameter is flagged as
“strong evidence” if group sizes exceed a minimum threshold, the CI excludes
zero, and $|z|$ exceeds a user-defined cutoff.

*Numeric parameters.* We fit a linear model $y = beta_0 + beta_1 x$ and report
the slope $beta_1$ (direction-aware) with its standard error and 95% CI. We
also report Spearman's rank correlation $rho$ (direction-aware) to capture
monotonic trends without linearity assumptions.

== Bootstrap evidence (per selected parameter)

For more robust, distribution-free diagnostics we bootstrap the selected
parameter:

- *Categorical:* resample the best and runner-up categories with replacement and
  compute the bootstrap distribution of $Delta bar(y)$. We report its mean,
  95% percentile CI, and the n-sigma score ($mu / sigma$), and visualise the
  histogram.
- *Numeric:* resample paired $(x, y)$ values and compute the bootstrap
  distribution of $beta_1$. We report the mean directional slope, percentile CI,
  n-sigma, and Spearman $rho$.

== Interaction heatmaps

For any parameter pair, numeric values are quantile-binned and the objective is
aggregated (mean or median). The resulting heatmap highlights pairwise
interactions and guards against misleading univariate conclusions.

== Importance and duplicates

We include Optuna's built-in fANOVA parameter importance when available and
flag duplicated configurations by hashing a user-selected parameter signature.

== Notes and limitations

Optuna trials are adaptive and therefore not i.i.d.; the reported $z$ and
n-sigma values are heuristic evidence rather than formal hypothesis tests.
Bootstrap intervals mitigate distributional assumptions but still require
sufficient, reasonably independent samples. For very small sample counts we
avoid strong-evidence flags.

// TODO: Update Optuna CSV export paths and sweep run IDs once the latest
// sweep export is finalized.
