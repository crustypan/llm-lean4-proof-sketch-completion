# Reproduce the final SketchFill archive

Run these commands in a separate checkout. The archives and integrity check require Python 3 and its standard library.

```sh
python3 docs/verify_archive.py
```

The check validates each distributed file, all JSONL row counts, the derived split separation, and pass@k for both completed comparisons. The results of the original research run are stored in [results/](../results/).

## Prepare the Lean judge and data

Install Elan and a compatible CUDA/PyTorch environment separately if generating completions. The archived judge pins Lean 4.15.0 and the exact Mathlib dependency revisions. The generation and training scripts import PyTorch, Transformers, PEFT, bitsandbytes, datasets, TRL and tqdm. The adapter records PEFT 0.18.0. Use compatible GPU package builds for your CUDA version.

```sh
mkdir -p work/data
for file in data/*.jsonl.gz; do
  name="$(basename "$file" .gz)"
  gzip -dc "$file" > "work/data/$name"
done
(cd lean && lake exe cache get && lake build)
export LEAN_PROJ="$PWD/lean"
export RUN_CONFIG_PATH="$PWD/configs/run_config.json"
```

The configuration uses the [official base model](https://huggingface.co/deepseek-ai/DeepSeek-Prover-V2-7B) and writes new artifacts under `work/`. To generate with the original run's weights, copy its saved `adapter_model.safetensors` into `work/adapter/` and compare it with the [adapter receipt](../configs/final-adapter-receipt.json). Restore the matching tokenizer and configuration from this archive.

```sh
mkdir -p work/adapter
gzip -dc configs/tokenizer.json.gz > work/adapter/tokenizer.json
cp configs/adapter_config.json configs/tokenizer_config.json configs/special_tokens_map.json configs/chat_template.jinja work/adapter/
```

The training script adds `<proof>` and `<eofill>` to the tokenizer. If training a fresh adapter, use `torchrun --standalone --nproc_per_node=3 src/train_stepwise_sft.py` with three suitable GPUs, or choose a compatible worker count. Training a new adapter can give different generation results.

## Replay or generate

For a GPU-free whole-file replay of the latest assembled candidates, run the original checker with the saved assembled Lean text. This produces a new result file.

```sh
mkdir -p work/lean-filtered
gzip -dc results/lean-filtered-generated_all.jsonl.gz > work/lean-filtered/generated_all.jsonl
GEN_JSONL="$PWD/work/lean-filtered/generated_all.jsonl" \
OUT_JSONL="$PWD/work/lean-filtered/verify_results.jsonl" JOBS=2 \
python3 src/run_lean_verify.py
```

For a fresh three-worker generation run, use a new output directory. The latest recorded run used one group attempt per sketch, up to 32 candidates per hole, immediate insertion, seed base 1234, and a 1,200-second per-hole Lean timeout. The source generator enables sampling when multiple candidates are requested.

```sh
run="$PWD/work/new-lean-filtered"
mkdir -p "$run"
export ADAPTER_DIR="$PWD/work/adapter"
export TOKENIZER_DIR="$ADAPTER_DIR"
SKETCH_JSONL="$PWD/work/data/sketch.raw.test.jsonl" OUT_JSONL="$run" \
K_PER_HOLE=32 N_ATTEMPTS=1 MAX_HOLES_PER_PROOF=320 MAX_NEW_TOKENS_PER_HOLE=512 \
IMMEDIATE_INSERT=1 GEN_SEED_BASE=1234 DO_SAMPLE=0 TEMPERATURE=0.7 TOP_P=0.95 \
HOLE_VERIFY_TIMEOUT=1200 torchrun --standalone --nproc_per_node=3 src/generate_lean_filtered.py
IN_DIR="$run" OUT_JSONL="$run/holes_all.jsonl" python3 src/merge_jsonl_shards.py
IN_JSONL="$run/holes_all.jsonl" OUT_JSONL="$run/generated_all.jsonl" \
SKIPPED_JSONL="$run/skipped_groups.jsonl" STRICT_COMPLETE=1 python3 src/assemble_holes.py
GEN_JSONL="$run/generated_all.jsonl" OUT_JSONL="$run/verify_results.jsonl" JOBS=2 \
python3 src/run_lean_verify.py
```

Sequential and parallel generators are also included. They use `K_ATTEMPTS` for full-sketch attempts and the same merge, assembly and verification scripts. The notebooks record the original invocation and use `original-layout/` and `local-environment/` placeholders for machine-specific locations. Use this guide for the organized checkout. The archive check and archived metrics were verified during publication. The GPU training and Lean compilation were not rerun for this archival import.
