# Reproduce SketchFill

Run the commands from the repository root. Write new outputs under `work/` so they can be compared with the archived results.

## Check the archive and the training data

The archive check uses Python 3 and its standard library. It checks file hashes, data and result counts, source-key overlap between splits, and pass@k for both completed comparisons.

```sh
python3 docs/verify_archive.py
```

The released corpus contains the source proof body, the sketch, and the ordered fragments removed from that proof. The following check reconstructs the proof bodies from the stored fragments.

```sh
python3 - <<'PY'
import gzip
import json

with gzip.open("data/Final_sketchdata.jsonl.gz", "rt", encoding="utf-8") as source:
    count = 0
    for line in source:
        record = json.loads(line)
        proof = record["sketch"]
        assert proof.count("<proof>") == len(record["fills"]), record["id"]
        for fragment in record["fills"]:
            proof = proof.replace("<proof>", fragment, 1)
        assert proof.strip() == record["proof"].strip(), record["id"]
        count += 1
print(f"Reconstructed {count} proof bodies")
PY
```

The expected count is 5,458. The archived data starts at the derived sketch stage. Recreating the initial proof-generation pool would require sampling the base model on the source benchmarks and screening those outputs with Lean. The original preprocessing notebooks document the masking and single-gap expansion. [Sketch extraction](../notebooks/sketch-FormalMATH.ipynb) · [Training examples](../notebooks/holewise_training.ipynb)

## Prepare the Lean judge

Install Elan and run the build commands in the pinned project directory. The project pins Lean 4.15.0 and the exact revisions of its Lean dependencies.

```sh
(cd lean && lake exe cache get && lake build)
export LEAN_PROJ="$PWD/lean"
```

## Replay the completed comparison

The README shows a five-candidate [sample replay](../README.md#2-recheck-saved-proofs-with-lean). A full replay rechecks 61,120 archived candidates. It uses a CPU and the saved candidate code. Adjust `JOBS` to suit the machine.

```sh
for method in sequential parallel; do
  run="$PWD/work/replayed-comparison/$method"
  mkdir -p "$run"
  gzip -dc "results/$method-verified-attempts.jsonl.gz" > "$run/candidates.jsonl"
  GEN_JSONL="$run/candidates.jsonl" OUT_JSONL="$run/verify_results.jsonl" \
    JOBS=2 TIMEOUT=1200 python3 src/run_lean_verify.py
done
```

For the reported scores, use the archived result check above. To compute pass@k from the new verdicts, run the [aggregation step](#compute-passk-from-new-verdicts).

## Prepare the model and data

GPU generation and training use CUDA and bfloat16-capable hardware. The scripts use PyTorch, Transformers, PEFT, bitsandbytes, datasets, TRL, and tqdm. Training also writes TensorBoard logs. Use builds compatible with your CUDA and driver versions. The adapter was saved with PEFT 0.18.0.

```sh
mkdir -p work/data
for file in data/*.jsonl.gz; do
  name="$(basename "$file" .gz)"
  gzip -dc "$file" > "work/data/$name"
done
export RUN_CONFIG_PATH="$PWD/configs/run_config.json"
```

### Use the original adapter

Restore the tokenizer and configuration, then place the original `adapter_model.safetensors` from the research storage in `work/adapter/`. The adapter receipt gives its size, SHA-256, and training step. [Adapter receipt](../configs/final-adapter-receipt.json)

```sh
mkdir -p work/adapter
gzip -dc configs/tokenizer.json.gz > work/adapter/tokenizer.json
cp configs/adapter_config.json configs/tokenizer_config.json \
  configs/special_tokens_map.json configs/chat_template.jinja work/adapter/
python3 - <<'PY'
import hashlib
import json
from pathlib import Path

receipt = json.loads(Path("configs/final-adapter-receipt.json").read_text())
with Path("work/adapter/adapter_model.safetensors").open("rb") as weights:
    digest = hashlib.sha256()
    for block in iter(lambda: weights.read(1024 * 1024), b""):
        digest.update(block)
assert digest.hexdigest() == receipt["sha256"], "Adapter SHA-256 mismatch"
print("Original adapter hash matched")
PY
export ADAPTER_DIR="$PWD/work/adapter"
export TOKENIZER_DIR="$ADAPTER_DIR"
```

### Train a fresh adapter from the released examples

Use this route if you have the released data and access to suitable GPUs. The training script adds the special tokens and saves a matching tokenizer. The example uses three GPUs as in the original run. The resulting adapter and subsequent candidates can differ from the archived run.

```sh
python3 - <<'PY'
import json
from pathlib import Path

config = json.loads(Path("configs/run_config.json").read_text())
config["out_dir"] = "work/retrained-adapter"
Path("work/retrain-config.json").write_text(json.dumps(config, indent=2) + "\n")
PY
RUN_CONFIG_PATH="$PWD/work/retrain-config.json" \
  torchrun --standalone --nproc_per_node=3 src/train_stepwise_sft.py
export ADAPTER_DIR="$PWD/work/retrained-adapter"
export TOKENIZER_DIR="$ADAPTER_DIR"
```

## Generate the sequential and parallel comparisons

Set `ADAPTER_DIR` and `TOKENIZER_DIR` using one of the two routes above. With the pinned Lean project ready, run the following from the repository root. The recorded settings use seed base 1234, a 512-token budget per hole, 32 attempts per sketch, temperature 0.7, and top-p 0.95. The generators enable sampling for repeated attempts. Only the sequential generator inserts earlier fills into the next prompt.

```sh
export SKETCH_JSONL="$PWD/work/data/sketch.raw.test.jsonl"
export K_ATTEMPTS=32 MAX_NEW_TOKENS_PER_HOLE=512 MAX_HOLES_PER_PROOF=320
export GEN_SEED_BASE=1234 DO_SAMPLE=0 TEMPERATURE=0.7 TOP_P=0.95
for method in sequential parallel; do
  run="$PWD/work/new-comparison/$method"
  mkdir -p "$run"
  IMMEDIATE_INSERT=1 OUT_JSONL="$run" \
    torchrun --standalone --nproc_per_node=3 "src/generate_${method}.py"
  IN_DIR="$run" OUT_JSONL="$run/holes_all.jsonl" python3 src/merge_jsonl_shards.py
  IN_JSONL="$run/holes_all.jsonl" OUT_JSONL="$run/generated_all.jsonl" \
    SKIPPED_JSONL="$run/skipped_groups.jsonl" STRICT_COMPLETE=1 python3 src/assemble_holes.py
  GEN_JSONL="$run/generated_all.jsonl" OUT_JSONL="$run/verify_results.jsonl" \
    JOBS=2 TIMEOUT=1200 python3 src/run_lean_verify.py
done
```

Use a fresh run directory for each repeat, especially when changing the number of workers. The merge step reads all the worker shards in that directory. The original run recorded a verdict for each sketch and attempt under both methods. [Sequential notebook](../notebooks/holewise_test_Sequential.ipynb) · [Parallel notebook](../notebooks/holewise_test_parallel.ipynb)

### Compute pass@k from new verdicts

Set `comparison` to `work/replayed-comparison` for the CPU replay or `work/new-comparison` for fresh generation. This checks that both methods have all 32 verdicts for every test sketch before scoring. Scores use the original Lean exit-code criterion.

```sh
comparison="work/replayed-comparison"
python3 - "$comparison" <<'PY'
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys

with gzip.open("data/sketch.raw.test.jsonl.gz", "rt", encoding="utf-8") as source:
    test_ids = {str(json.loads(line)["id"]) for line in source}

for method in ("sequential", "parallel"):
    verdicts = defaultdict(dict)
    path = Path(sys.argv[1]) / method / "verify_results.jsonl"
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            case = str(row.get("rid") or row["id"])
            attempt = int(row["gen_attempt"])
            if attempt in verdicts[case]:
                raise ValueError(f"{method}: duplicate attempt {attempt} for {case}")
            verdicts[case][attempt] = row["pass"]
    if set(verdicts) != test_ids or any(set(rows) != set(range(32)) for rows in verdicts.values()):
        raise ValueError(f"{method}: incomplete test coverage; inspect the skipped groups and worker logs")
    for k in (1, 12, 24, 32):
        passed = sum(any(rows[i] for i in range(k)) for rows in verdicts.values())
        print(f"{method} pass@{k} = {passed}/{len(test_ids)} ({passed / len(test_ids):.2%})")
PY
```

## Replay or repeat the per-gap Lean check

The archived follow-up contains partial-check and assembly records. For a CPU-only whole-file replay of the assembled candidates, use the following command after preparing the Lean judge.

```sh
run="$PWD/work/replayed-lean-filtered"
mkdir -p "$run"
gzip -dc results/lean-filtered-generated_all.jsonl.gz > "$run/generated_all.jsonl"
GEN_JSONL="$run/generated_all.jsonl" OUT_JSONL="$run/verify_results.jsonl" \
  JOBS=2 TIMEOUT=1200 python3 src/run_lean_verify.py
```

For a fresh GPU run, set the model and data paths as above. This experiment has one group attempt per sketch and up to 32 candidate fragments per hole.

```sh
run="$PWD/work/new-lean-filtered"
mkdir -p "$run"
SKETCH_JSONL="$PWD/work/data/sketch.raw.test.jsonl" OUT_JSONL="$run" \
  K_PER_HOLE=32 N_ATTEMPTS=1 MAX_HOLES_PER_PROOF=320 MAX_NEW_TOKENS_PER_HOLE=512 \
  IMMEDIATE_INSERT=1 GEN_SEED_BASE=1234 DO_SAMPLE=0 TEMPERATURE=0.7 TOP_P=0.95 \
  HOLE_VERIFY_TIMEOUT=1200 torchrun --standalone --nproc_per_node=3 src/generate_lean_filtered.py
IN_DIR="$run" OUT_JSONL="$run/holes_all.jsonl" python3 src/merge_jsonl_shards.py
IN_JSONL="$run/holes_all.jsonl" OUT_JSONL="$run/generated_all.jsonl" \
  SKIPPED_JSONL="$run/skipped_groups.jsonl" STRICT_COMPLETE=1 python3 src/assemble_holes.py
GEN_JSONL="$run/generated_all.jsonl" OUT_JSONL="$run/verify_results.jsonl" \
  JOBS=2 TIMEOUT=1200 python3 src/run_lean_verify.py
```

The 739 archived outputs are assembly records. To count whole-proof successes over the original 955 sketches, use the new verifier results and account for cases that never reached assembly. [Follow-up results and limitations](results.md#latest-lean-filtered-run)
