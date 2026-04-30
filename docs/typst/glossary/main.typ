#import "@preview/glossarium:0.5.10": make-glossary
#import "../shared/glossary.typ": register-aria-glossary, print-aria-glossary

#set document(title: "ARIA-NBV Glossary")
#set page(paper: "a4", margin: (x: 22mm, y: 20mm))
#set text(font: "New Computer Modern", size: 10pt)
#show: make-glossary

#register-aria-glossary()

= ARIA-NBV Glossary

This document renders the canonical Typst glossary source used by the Quarto
docs, Typst papers/slides, and KG export.

#print-aria-glossary(show-all: true, disable-back-references: true)
