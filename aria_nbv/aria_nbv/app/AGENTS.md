---
scope: module
applies_to: aria_nbv/aria_nbv/app/**
summary: Streamlit app, panel, state, and UI guidance for the Aria-NBV inspection surfaces.
---

# App Guidance

Follow [../../AGENTS.md](../../AGENTS.md) plus this file for work under
`aria_nbv/aria_nbv/app/`. Durable app ownership notes live in [README.md](README.md).

## Rules
- Keep `app.py` and panel entrypoints thin; domain logic belongs in package
  modules such as data handling, pose generation, rendering, RRI, or VIN.
- Model app-facing state through typed state/config objects instead of free-form
  Streamlit session-state access.
- Panels should compose shared plotting helpers and builders, not implement
  plotting math inline.
- Preserve display-only frame corrections as UI concerns; do not let them leak
  into cache, model, rendering, or training semantics.

## Verification
- Run targeted app/panel tests for changed panels or app state.
- Run figure-construction smoke tests when app work changes Plotly traces,
  layout, or shared plotting helper use.
