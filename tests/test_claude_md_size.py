"""Keep CLAUDE.md inside Claude Code's instruction-file budget.

Claude Code caps all instruction files loaded into a session (the user's global CLAUDE.md
and its imports plus this repo's CLAUDE.md) at 150,000 characters in total. The global
files take about 60k, so this repo's file must stay well under the remainder. Commands go in
docs/commands.md and design detail in docs/architecture.md; CLAUDE.md keeps the rules and
one-line pointers.
"""

from pathlib import Path

CLAUDE_MD = Path(__file__).resolve().parents[1] / "CLAUDE.md"
MAX_CHARS = 50_000


def test_claude_md_within_budget():
    n = len(CLAUDE_MD.read_text(encoding="utf-8"))
    assert n <= MAX_CHARS, (
        f"CLAUDE.md is {n:,} chars (limit {MAX_CHARS:,}). Move command detail to "
        "docs/commands.md and findings to docs/architecture.md or a study doc."
    )
