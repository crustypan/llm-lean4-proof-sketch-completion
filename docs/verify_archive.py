"""Verify the archived B4 files and recompute reported counts."""

from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_rows(relative):
    with gzip.open(ROOT / relative, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def verify_files():
    manifest = (ROOT / "docs/SHA256SUMS").read_text().splitlines()
    for line in manifest:
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {name}")
    return len(manifest)


def benchmark(method):
    cases = defaultdict(dict)
    rows = passed = sorry_diagnostics = 0
    for row in read_rows(f"results/{method}-verified-attempts.jsonl.gz"):
        case = str(row["rid"] if row["rid"] is not None else row["id"])
        attempt = row["gen_attempt"]
        if attempt in cases[case]:
            raise ValueError(f"Duplicate attempt {attempt} in {method}/{case}")
        cases[case][attempt] = row["pass"]
        rows += 1
        passed += row["pass"]
        sorry_diagnostics += row["pass"] and row["sorry_diagnostic"]
    for case, attempts in cases.items():
        if set(attempts) != set(range(32)):
            raise ValueError(f"Incomplete attempt set in {method}/{case}")
    result = {
        "attempts": rows,
        "passed_attempts": passed,
        "cases": len(cases),
        "passed_attempts_with_sorry_diagnostic": sorry_diagnostics,
        "pass_at_k": {
            str(k): sum(any(attempts[i] for i in range(k)) for attempts in cases.values())
            for k in (1, 12, 24, 32)
        },
    }
    return set(cases), result


def main():
    count = verify_files()
    expected = json.loads((ROOT / "results/summary.json").read_text())
    for relative, rows in expected["rows"].items():
        if sum(1 for _ in read_rows(relative)) != rows:
            raise ValueError(f"Row count mismatch for {relative}")
    groups = {}
    source_benches = Counter()
    for split in ("test", "train", "valid"):
        filename = f"sketch.raw.{split}.jsonl"
        keys = set()
        for row in read_rows(f"data/{filename}.gz"):
            key = (str(row["source_bench"]), str(row["source_id"]))
            keys.add(key)
            source_benches[key[0]] += 1
        groups[filename] = keys
    overlap = {
        f"{left} vs {right}": len(groups[left] & groups[right])
        for i, left in enumerate(groups) for right in list(groups)[i + 1:]
    }
    if overlap != expected["split_overlap_by_source_bench_and_id"] or source_benches != expected["input_source_benches"]:
        raise ValueError("Source split mismatch")
    case_ids = []
    for method in ("sequential", "parallel"):
        ids, counts = benchmark(method)
        case_ids.append(ids)
        if counts != expected["verified_comparators"][method]:
            raise ValueError(f"Benchmark mismatch for {method}")
        print(method, json.dumps(counts["pass_at_k"]))
    if case_ids[0] != case_ids[1]:
        raise ValueError("Comparator cases differ")
    generated = list(read_rows("results/lean-filtered-generated_all.jsonl.gz"))
    holes = list(read_rows("results/lean-filtered-holes_all.jsonl.gz"))
    skipped = list(read_rows("results/lean-filtered-skipped_groups.jsonl.gz"))
    latest = {
        "generated_all.jsonl": {
            "rows": len(generated), "complete_hole_groups": sum(bool(x["holes_complete"]) for x in generated),
            "generated_still_has_hole": sum(bool(x["gen_still_has_hole"]) for x in generated),
        },
        "holes_all.jsonl": {
            "rows": len(holes), "passing_candidate_total": sum(int(x["passing_candidates"]) for x in holes),
            "tested_candidate_total": sum(int(x["tested_candidates"]) for x in holes),
        },
        "skipped_groups.jsonl": {"rows": len(skipped)},
    }
    if latest != expected["latest_lean_filtered_run"]:
        raise ValueError("Lean-filtered generation counts differ")
    print(f"Verified {count} files, {len(case_ids[0])} matched cases, and {len(generated)} assembled outputs.")


if __name__ == "__main__":
    main()
