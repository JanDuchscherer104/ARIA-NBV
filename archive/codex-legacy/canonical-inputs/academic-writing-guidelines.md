# Academic Writing Guidelines (Research Notes)

## Why this note exists
- Collects external guidance for what to enforce in `.codex/AGENTS-paper-slides.md` when writing the final paper and slides.

## Sources consulted (primary)
- Mensh & Kording (2017), “Ten Simple Rules for Structuring Papers”: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005619
- Gopen & Swan (1990), “The Science of Scientific Writing” (PDF mirror): https://writingcenter.tufts.edu/wp-content/uploads/ScienceofScientificWriting.pdf
- Purdue OWL (IEEE format + abstract requirements): https://owl.purdue.edu/owl/research_and_citation/ieee_style/ieee_overview.html
- IEEE Author Center (ethics): https://ieeeauthorcenter.ieee.org/about-ieee-author-center/author-ethics/
- ICMJE (authorship criteria + AI-assisted technologies): https://www.icmje.org/recommendations/browse/roles-and-responsibilities/defining-the-role-of-authors-and-contributors.html
- NeurIPS paper checklist (reproducibility template): https://neurips.cc/public/guides/PaperChecklist
- IEEE Author Center (figures/tables basics): https://ieeeauthorcenter.ieee.org/create-your-ieee-article/create-graphics-for-your-article/using-figures-tables-and-parts/

## Key takeaways to encode as agent instructions

### Paper-level structure (story)
- One “main claim”/contribution, reflected consistently in title, abstract, intro, results, conclusion.
- Use “Context → Content → Conclusion” at multiple scales (paper, section, paragraph).
- Related work is comparative (positioning + trade-offs), not a dump of citations.

### Clarity at the sentence/paragraph level
- Lead paragraphs with what they are about (topic); end with what to remember (takeaway).
- Keep subject and verb close; put action in verbs; prefer concrete subjects (avoid abstract noun chains).
- Prefer active voice when it clarifies responsibility and reduces ambiguity.

### Abstract hygiene (IEEE-style guidance)
- One paragraph, self-contained, 150–250 words; avoid abbreviations, references/footnotes; no displayed equations/tables.
- Must cover: topic, purpose, methods, results, conclusion.

### Evidence and reporting (especially for ML/CV)
- Every claim needs matching evidence (measurement, table, figure); avoid causal claims without ablations.
- Report enough experimental detail to replicate: datasets/splits, preprocessing, hyperparameters, seeds, eval protocol.
- Prefer uncertainty estimates: multiple runs, CI/error bars, statistical tests where appropriate.
- Track compute/infrastructure if it impacts feasibility or is part of the contribution.

### Reproducibility and transparency
- Make explicit what is released (code/configs/scripts) and what isn’t (and why).
- Provide sufficient detail for a skilled reader to re-implement critical parts.
- Document dataset and asset licenses; cite and attribute reused figures.

### Research integrity, authorship, and AI tools
- Follow IEEE ethics: originality, proper citation, no plagiarism/duplicate submissions, no deceptive figure manipulation.
- Follow ICMJE-style guidance: AI tools are not authors; humans remain responsible and should disclose AI use if required by the venue.

## NBV-project-specific suggestions
- Maintain a “Notation” mini-table early in Method (symbols, frames, shapes/units) to keep RRI/pose notation consistent.
- Ensure every figure caption is stand-alone (what/setting/key takeaway) and references the metric definition (accuracy/completeness/RRI).
- Keep a running “Assumptions + Limitations” list and promote it into a Limitations section for the final draft.

