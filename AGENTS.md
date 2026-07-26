# Repository guidance

This is a learning lab. The code is the teaching artifact, so readability,
explicit tensor shapes and falsifiable checks matter more than cleverness.

Read `CLAUDE.md` for the local Python/Jupyter environment, commands and
non-obvious repository invariants; those instructions apply to every coding
agent despite the filename.

Before adding or substantially changing a lesson, read
`.claude/skills/add-lesson/SKILL.md`. Follow its module → tests → notebook/page
workflow, then run:

```bash
make lessons
make check
```

If a notebook changed, also execute it top to bottom with
`make notebooks` (or target it through `tools/nbuild.py --run`). Never commit
notebook outputs.

Preserve the offline `file://` contract for `interactive/`: no CDN, `fetch()`,
XHR or hidden network dependency. Use the shared CSS/JavaScript primitives and
keep every numerical claim available as a table or test.
