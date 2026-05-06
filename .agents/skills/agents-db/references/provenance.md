# Agents DB Provenance

Every active issue and todo needs enough context for a future agent to verify
why it exists without chat history.

## Reference Prefixes

- `repo:<path>#<anchor-or-section>` for repo files, docs, code, tests, skills,
  or generated context.
- `bib:<citation-key>` for papers in `docs/references.bib`.
- `arxiv:<id>`, `doi:<doi>`, or `s2:<paperId>` for durable paper identifiers.
- `url:<https-url>` for external API or tool documentation.
- `context7:<library-id>` for Context7-resolved external library docs.
- `litkg:<profile-or-command>` for litkg-rs context-pack, capability, or KG
  command evidence.

## Litkg Use

For broad or literature-backed DB additions, run a litkg route/query first and
copy only the relevant source pointers into `references`. Prefer stable source
pointers over long summaries.

Backlog records should remain compact but auditable. If a context pack omits
active backlog rows or references, amend `issue-023` or `issue-025` rather than
inventing a parallel tracker.
