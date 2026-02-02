# Make Context Error 127 Fix

## Summary
- Investigated `make context` failure (`Error 127`), traced to missing external commands (`rg` or `tree`) invoked by context index and tree output steps.
- Added fallbacks so context generation completes even when `rg` or `tree` are not installed.

## Changes
- `.codex/skills/aria-nbv-context/scripts/nbv_context_index.sh`: added `rg` availability check and `find` fallback for file listing and counts.
- `Makefile`: added fallback to `find` when `tree` is unavailable for directory listing.
- `.codex/AGENTS_INTERNAL_DB.md`: documented the gotcha and new fallbacks.

## Verification
- Ran `make context` successfully after changes.

## Follow-ups
- If `syrenka` is missing in the venv, `make context` will still fail (module import error). Consider adding a preflight check with a clearer message if this becomes a recurring issue.
