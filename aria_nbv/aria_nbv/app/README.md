# App Ownership

`aria_nbv.app` owns Streamlit inspection and diagnostic surfaces for cached
snippets, candidates, RRI bins, VIN outputs, and planning/RL scaffolds.

The app is not the owner of data contracts, model semantics, rendering math, or
training behavior. Those contracts should stay in their package modules and be
composed by panels through typed state and shared plotting helpers.
