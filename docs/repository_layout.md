# Repository layout

Reusable application/library code: `brax/`.

Standalone utilities belong in `scripts/`, grouped by purpose. Tests belong in
`tests/`, with subsystem folders where needed. Keep generated videos, figures,
logs, checkpoints, and datasets outside source and test directories.

| Organized location | Files relocated |
| --- | ---: |
| `tests/` | 74 |

Run Python commands from the repository root with the appropriate environment
activated. Relocated scripts also locate the repository themselves when executed
by filename. Use `python -m pytest <test-directory>` to run a selected suite.

External launchers must use the new paths. The versioned [path map](reorganization.json) records old and new source locations.
Pre-edit backups are retained locally in the scratch workspace.

Reusable rendering functions stay in the library; standalone GIF/video generation
belongs with the tools. Existing datasets, run snapshots, dependency checkouts,
and user edits are preserved.
