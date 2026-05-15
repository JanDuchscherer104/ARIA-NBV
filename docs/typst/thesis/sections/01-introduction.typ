#import "../../shared/macros.typ": *

= Introduction

#gls("next-best-view") planning selects future sensor views to improve a reconstruction under a finite acquisition budget. ARIA-NBV focuses this problem on egocentric indoor trajectories, where candidate views should be ranked by expected reconstruction quality instead of by geometry-only coverage proxies.

The thesis starts from the seminar-paper formulation of #gls("relative-reconstruction-improvement")\-driven #gls("next-best-view", first: false) and extends it toward a thesis-scale treatment of quality-driven view selection, model calibration, and planning behavior.
