# Manuscript integration, 31 October 2026

This copy of the standalone research package supports the revised manuscript.
The README's statements that the paper was unchanged describe the preceding
research stage. The current revision integrates the scoped theory, controlled
figure and tables, and prior-work comparison. The source records and frozen
backend modules are unchanged. RESEARCH_STAGE_SHA256.json preserves the earlier
package manifest; SHA256.json describes this integrated copy.

Regenerate the new figure and tables from the recorded data with:

```sh
python -B manuscript_assets.py --latex /path/to/LaTeX
```

The archive rule in the paper project's AGENTS.md applies before replacing
existing manuscript files or rebuilding the active paper. Render to a separate
working directory when inspecting assets. The old range-baseline artifact is
not modified by this package.
