#import "@preview/charged-ieee:0.1.4": ieee


// #import "@preview/muchpdf:0.1.1": muchpdf

#let shared_path = "/_shared/"
#let figures_path = shared_path + "figures/"

#show: ieee.with(
  title: [A Typesetting System to Untangle the Scientific Writing Process],
  abstract: [
    The process of scientific writing is often tangled up with the intricacies of typesetting, leading to frustration and wasted time for researchers. In this paper, we introduce Typst, a new typesetting system designed specifically for scientific writing. Typst untangles the typesetting process, allowing researchers to compose papers faster. In a series of experiments we demonstrate that Typst offers several advantages, including faster document creation, simplified syntax, and increased ease-of-use.
  ],
  authors: (
    (
      name: "Jan Duchscherer",
      department: [Computer Science & Mathematics],
      organization: [Munich University of Applied Sciences],
      location: [Munich, Germany],
      email: "duchsche@hm.edu"
    ),
  ),
  index-terms: ("Scientific writing", "Typesetting", "Document creation", "Syntax"),
  bibliography: bibliography(shared_path + "references.bib"),
  figure-supplement: [Fig.],
)


#include "sections/example-content.typ"