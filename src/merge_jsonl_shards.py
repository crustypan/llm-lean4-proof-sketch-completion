
import os, glob
from pathlib import Path

def main():
    in_dir = os.environ["IN_DIR"]
    out_jsonl = os.environ["OUT_JSONL"]

    files = sorted(glob.glob(str(Path(in_dir) / "gen_rank*.jsonl")))
    assert files, f"no shard files found in: {in_dir}"

    outp = Path(out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)

    cnt = 0
    with open(outp, "w", encoding="utf-8") as w:
        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    w.write(ln + "\n")
                    cnt += 1
    print(f"merged {len(files)} shards -> {outp}  (lines={cnt})")

if __name__ == "__main__":
    main()

