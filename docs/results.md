# Reading the saved experiments

## Latest Lean-filtered run

The `20260213-112100` run generated several candidates per hole and tried them with Lean in the current partial sketch. Earlier fills were visible to subsequent holes. The merge and assembly stages retained 739 complete groups and recorded 216 skipped groups across 955 input sketches. The per-hole ledger contains 1,986 rows and reports 13,831 passing candidate checks out of 23,789 tested. These checks ran on partial sketches, which could still contain `sorry` in other holes.

| Archive | Rows | Purpose |
| --- | ---: | --- |
| [Generated files](../results/lean-filtered-generated_all.jsonl.gz) | 739 | Assembled Lean text and generation metadata |
| [Hole ledger](../results/lean-filtered-holes_all.jsonl.gz) | 1,986 | Candidate counts and chosen fills |
| [Skipped groups](../results/lean-filtered-skipped_groups.jsonl.gz) | 216 | Missing or duplicate hole indices recorded at assembly |

The archived source directory for this run had no whole-file `verify_results.jsonl`. The later analysis cells in the original Lean-filtered notebook point to the completed sequential comparison's log. The table on the repository homepage attributes those values to the sequential comparison.

## Completed comparison

Each compact ledger has 30,560 rows, one row for each case and zero-based attempt index. Both ledgers contain the same 955 cases with attempts 0 through 31. The `pass` field retains the original return-code criterion. The derived `sorry_diagnostic` flag detects the saved Lean diagnostic that a declaration uses `sorry`. The console tail can be truncated, and this flag does not constitute an axiom audit.

| Ledger | Successful exits | Successful exits with a saved `sorry` diagnostic |
| --- | ---: | ---: |
| [Sequential](../results/sequential-verified-attempts.jsonl.gz) | 24,701 | 1,269 |
| [Parallel](../results/parallel-verified-attempts.jsonl.gz) | 23,156 | 1,350 |

The archived generator scripts are implementation snapshots for each method. The compact ledgers preserve the attempted Lean code and its observed status for the finished comparisons. [Computed counts](../results/summary.json) · [Source manifest](source-manifest.json)

## Training records

The recorded final adapter completed one epoch and 463 training steps. The saved validation mean token accuracy is 0.9309, a token-level quantity separate from proof checking. [Training history](../results/log_history.json) · [Validation metrics](../results/metrics_eval.json)
