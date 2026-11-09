import json, jsonlines, re
from pathlib import Path
from hashlib import md5

# ---------- 读取一条记录里的 Lean 完整代码 ----------
def get_lean_code(rec: dict) -> str | None:
    for k in ("lean_block", "leancode", "lean_code", "code", "proof"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v
    # 有些数据把 header 与 body 分开存
    hdr = (rec.get("header") or "").strip()
    body = (rec.get("body") or rec.get("generation") or rec.get("attempt") or "").strip()
    if hdr and body:
        return f"{hdr}\n\n{body}"
    return None

THEOREM_RE = re.compile(r'(?ms)^\s*(theorem|lemma)\s+.*?:=\s*by\b')

# 抓 have/suffices 一行式声明（保留语句，抹掉证明体）
HAVE_RE = re.compile(r'^\s*have\b([^:\n]*?:)?\s*(.+?)\s*:=\s*by\b', re.UNICODE)
SUFF_RE = re.compile(r'^\s*suffices\b\s*(.+?)\s*:=\s*by\b', re.UNICODE)

def make_sketch_from_lean(lean: str) -> dict:
    # 1) 切头尾
    m = THEOREM_RE.search(lean)
    if not m:
        # 找不到 theorem/lemma，就整体做一个占位
        head, body = "", lean
    else:
        pos = m.end()
        head, body = lean[:pos], lean[pos:]

    lines = body.splitlines()
    steps = []
    sketch_lines_train = []      # 训练版： -- <proof>
    sketch_lines_compile = []    # 可编译版： by sorry
    gap_id = 1

    def add_gap(line_show: str, kind: str, stmt: str | None):
        nonlocal gap_id
        # 训练版
        if line_show.strip().endswith("by"):
            l1 = f"{line_show} -- <proof>"
            l2 = f"{line_show} sorry"
        else:
            # 比如 calc、simp 等无显式 by 的行，补一行注释
            l1 = line_show
            l2 = line_show
            sketch_lines_train.append(l1)
            sketch_lines_train.append("  -- <proof>")
            sketch_lines_compile.append(l2)
            sketch_lines_compile.append("  by sorry")
            steps.append({"idx": gap_id, "kind": kind, "stmt": (stmt or "").strip()})
            gap_id += 1
            return

        sketch_lines_train.append(l1)
        sketch_lines_compile.append(l2)
        steps.append({"idx": gap_id, "kind": kind, "stmt": (stmt or "").strip()})
        gap_id += 1

    for raw in lines:
        line = raw.rstrip()

        # 2) have : ... := by
        mh = HAVE_RE.match(line)
        if mh:
            # mh.group(2) 是语句部分
            stmt = mh.group(2)
            # 只保留到 "by" 之前
            cut = line.split(":= by")[0] + ":= by"
            add_gap(cut, kind="have", stmt=stmt)
            continue

        # 3) suffices ... := by
        ms = SUFF_RE.match(line)
        if ms:
            stmt = ms.group(1)
            cut = line.split(":= by")[0] + ":= by"
            add_gap(cut, kind="suffices", stmt=stmt)
            continue

        # 4) calc / refine / cases / rcases / obtain 等视作一步
        if re.match(r'^\s*(calc|refine|cases|rcases|obtain)\b', line):
            # 尽可能展示原行，gap 放下一行
            sketch_lines_train.append(line)
            sketch_lines_compile.append(line)
            sketch_lines_train.append("  -- <proof>")
            sketch_lines_compile.append("  by sorry")
            steps.append({"idx": gap_id, "kind": "block", "stmt": ""})
            gap_id += 1
            continue

        # 5) 常见直接战术（simp/linarith/ring 等）——也视作一步
        if re.match(r'^\s*(simp|aesop|linarith|nlinarith|ring|omega|tautology|tactic)\b', line):
            sketch_lines_train.append("-- <proof>  " + line.strip())
            sketch_lines_compile.append("by sorry  -- " + line.strip())
            steps.append({"idx": gap_id, "kind": "tactic", "stmt": ""})
            gap_id += 1
            continue

        # 默认原样抄过去
        sketch_lines_train.append(line)
        sketch_lines_compile.append(line)

    # 如果什么结构都没抓到，直接做“单块占位”
    if not steps:
        sketch_lines_train = ["  -- <proof>"]
        sketch_lines_compile = ["  by sorry"]

    sketch_train = (head + "\n" + "\n".join(sketch_lines_train)).strip() if head else "\n".join(sketch_lines_train).strip()
    sketch_compile = (head + "\n" + "\n".join(sketch_lines_compile)).strip() if head else "\n".join(sketch_lines_compile).strip()

    return {
        "sketch_text": sketch_train,
        "sketch_compilable": sketch_compile,
        "num_gaps": len(steps),
        "steps": steps
    }

def main(in_path="data.jsonl", out_path="sketch.jsonl"):
    in_path, out_path = Path(in_path), Path(out_path)
    assert in_path.exists(), f"{in_path} 不存在"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out = jsonlines.open(out_path, "w")
    n_in, n_ok = 0, 0

    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            n_in += 1
            rec = json.loads(line)
            code = get_lean_code(rec)
            if not code:
                continue

            sk = make_sketch_from_lean(code)
            rid = str(rec.get("id") or rec.get("name") or md5(code.encode()).hexdigest()[:12])

            out.write({
                "id": rid,
                "sketch_text": sk["sketch_text"],
                "sketch_compilable": sk["sketch_compilable"],
                "num_gaps": sk["num_gaps"],
                "steps": sk["steps"],
                "full_proof": code
            })
            n_ok += 1

    out.close()
    print(f"done: input={n_in}, converted={n_ok}, written -> {out_path}")

if __name__ == "__main__":
    main()