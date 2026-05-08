#!/usr/bin/env bash
# Make .agents/skills/<name> discoverable to Claude Code by symlinking each
# skill into .claude/skills/<name>. Claude activates skills by matching the
# user's task against the SKILL.md `description` field; the existing
# .agents/skills/*/SKILL.md descriptions are written to be cross-vendor and
# Claude reads them transparently.
#
# This script is idempotent: it removes stale .claude/skills/<name> symlinks
# whose source no longer exists and creates missing ones. It refuses to clobber
# a real (non-symlink) directory under .claude/skills/.
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root"

src=".agents/skills"
dst=".claude/skills"

if [ ! -d "$src" ]; then
    echo "sync-claude-skills: missing $src" >&2
    exit 1
fi
mkdir -p "$dst"

# Remove stale entries first.
shopt -s nullglob
for entry in "$dst"/*; do
    name="$(basename "$entry")"
    if [ ! -e "$src/$name/SKILL.md" ]; then
        if [ -L "$entry" ]; then
            rm "$entry"
            echo "removed stale: $entry"
        else
            echo "sync-claude-skills: refusing to remove non-symlink $entry" >&2
        fi
    fi
done

# Create or refresh symlinks.
created=0
skipped=0
for skill_md in "$src"/*/SKILL.md; do
    skill_dir="$(dirname "$skill_md")"
    name="$(basename "$skill_dir")"
    target="$dst/$name"
    relative_src="../../$skill_dir"
    if [ -L "$target" ]; then
        current="$(readlink "$target")"
        if [ "$current" = "$relative_src" ]; then
            skipped=$((skipped + 1))
            continue
        fi
        rm "$target"
    elif [ -e "$target" ]; then
        echo "sync-claude-skills: $target exists and is not a symlink; skipping" >&2
        continue
    fi
    ln -s "$relative_src" "$target"
    created=$((created + 1))
    echo "linked: $target -> $relative_src"
done

echo "sync-claude-skills: $created created, $skipped already current"
