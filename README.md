# SketchFill · Gap Filling in Lean 4

This repository archives the final B4 graduation research on completing missing parts of a Lean proof. SketchFill inserts `<proof>` markers into proof sketches, trains a DeepSeek-Prover-V2-7B adapter to fill one hole at a time, and assembles the fills into Lean files. The final training variant is `holewise_training_2`. The last recorded Lean-filtered generation run is `20260213-112100`.

## Final results

The saved Lean-filtered sequential run assembled **739 of 955** input sketches and skipped **216** groups with missing fills. The archive contains 1,986 per-hole records with 23,789 candidate checks. A whole-file verifier output for this run was absent from its result directory. The generation count therefore describes assembly coverage.

The completed comparison used the same 955 sketch cases with 32 attempts per case. Pass@k is the fraction of cases with at least one recorded successful Lean exit among their first k attempts.

| Completed benchmark | Pass@1 | Pass@12 | Pass@24 | Pass@32 |
| --- | ---: | ---: | ---: | ---: |
| Sequential filling | 775/955 (81.15%) | 888/955 (92.98%) | 904/955 (94.66%) | 908/955 (95.08%) |
| Parallel filling | 727/955 (76.13%) | 865/955 (90.58%) | 885/955 (92.67%) | 892/955 (93.40%) |

The original verifier counts exit status zero. It recorded a `sorry` diagnostic in 1,269 of 24,701 successful sequential attempts and 1,350 of 23,156 successful parallel attempts. The attempt records keep this flag for inspection. [Results summary](results/summary.json) · [Verification notes](docs/results.md)

## Files

| Path | Contents |
| --- | --- |
| [src/](src/) | Final adapter training, per-hole and whole-file generation scripts, merge, assembly, and Lean verification |
| [configs/](configs/) | Training and adapter configurations, matching tokenizer, and final weights' SHA-256 and source location |
| [data/](data/) | Derived corpus, train/validation/test sketches, and per-hole training targets |
| [results/](results/) | Final generation and skip records, both full-file comparison ledgers, and training metrics |
| [notebooks/](notebooks/) | Original data preparation, training, evaluation, and analysis notebooks with execution outputs cleared |
| [lean/](lean/) | Lean 4.15.0 judge scaffold and pinned Mathlib dependencies |
| [docs/](docs/) | Reproduction instructions, result definitions, and source file hashes |
| [history/](history/) | Earlier Gap Filling source snapshots and recorded check summaries |

The data contains 4,191 training, 312 validation, and 955 test sketch records derived from FormalMATH and miniF2F. The splits share zero source keys when grouped by benchmark and source ID. The final training run contains 11,102 per-hole training rows and 835 validation rows. [FormalMATH](https://github.com/Sphere-AI-Lab/FormalMATH-Bench) · [miniF2F](https://github.com/facebookresearch/miniF2F) · [DeepSeek-Prover-V2-7B](https://huggingface.co/deepseek-ai/DeepSeek-Prover-V2-7B)

## Reproduce

From the repository root, run `python3 docs/verify_archive.py` to validate the archived file checksums and recompute the reported counts and pass@k values. GPU generation and Lean replay use the pinned Lean scaffold and the original adapter identified by the [adapter receipt](configs/final-adapter-receipt.json). The adapter weights remain in the original research storage. [Full reproduction guide](docs/reproduce.md)

[File-level source provenance](docs/source-manifest.json) records the original and archived SHA-256 for each selected source or data file. The comparison ledgers keep the generated Lean code, attempt order and verdict and omit machine-local paths and raw console tails.
