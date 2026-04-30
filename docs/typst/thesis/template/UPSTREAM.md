# Upstream provenance

- Upstream URL: https://github.com/ls1intum/thesis-template-typst
- Pinned commit: `53181fa054565415af9e7304e22308c36ace7369`
- Pinned tag: `v1.0.31`
- Import date: 2026-04-30
- Import mode: vendored source snapshot; not a git submodule, not a fork.
- License note: upstream is MIT licensed. The upstream `LICENSE` file is preserved in this directory.

## Imported surfaces

- Thesis layout support, adapted locally through `layout/thesis_template.typ`.
- Proposal layout support, adapted locally through `layout/proposal_template.typ`.
- Shared layout utilities required by those two document types.

## Local adaptations planned

- Replace TUM cover/title-page branding with HM branding.
- Keep ARIA-NBV bibliography, glossary, and notation imports available to thesis and proposal chapters.
- Keep the template locally editable while preserving this provenance record for future upstream refreshes.
- Avoid copying upstream sample content, CI workflows, registration forms, feedback logs, and bundled fonts unless thesis requirements later need them.
