# Reviewed raw evidence package (2026-11-24)

This package contains original runtime logs and action/reference records linked to experiments with a `review.json` and `experiment.json` in the D-drive project workspace. Records with `review.json: passed=false` are included as reviewed failures and clearly labeled; experiments without a review record are excluded. Rendered PNG/MP4 artifacts are excluded because the presentation/video evidence is already published separately.

## Contents

- `runtime-runs/`: one ZIP per complete D-drive runtime directory, with all files in that run preserved under `runs/<run-id>/`. Only runs referenced by reviewed trials are included. See `manifests/runtime-run-archives.csv` for paths, sizes, hashes, verdicts, and linked trials.
- `action-data/reviewed-experiment-records-01-of-12.zip` through `12-of-12.zip`: non-rendered review, experiment, action input, preflight, metrics, and raw per-trial logs from all reviewed experiment directories. Original workspace-relative paths are preserved under `workspace/`; `manifests/reviewed-sidecar-archives.csv` maps each part to its trial directories.
- `action-data/cmu-13_30-motion-source.zip`: original CMU ASF/AMC clip, local retargeted skeleton motion NPZ, license and provenance, with attribution and non-resale notice.
- `manifests/reviewed-sidecar-files.csv`: file-level SHA-256 manifest for the records in the sidecar archive.

## Coverage limits

The D-drive workspace has 195 reviewed experiment directories (115 passed, 80 failed). Of their runtime references, 57 complete runtime directories are physically available in this D-drive checkout and included here. 118 distinct older runtime references point to `/home/wangrunqi/sonic_takeover_share_20261114/...`, which is absent from the D-drive checkout; their review/action sidecars are included where present, but the missing old runtime trees are not represented as uploaded. Twenty review records have no usable runtime path. No unreviewed experiment is included.

This archive preserves measured data; inclusion does not turn failed or exploratory results into acceptance claims.
