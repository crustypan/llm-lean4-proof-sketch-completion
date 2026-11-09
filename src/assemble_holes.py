
# assemble_holes.py
import os, json, re
from collections import defaultdict
from pathlib import Path

HOLE = "<proof>"

def normalize_body_for_by(block: str) -> str:
    """保证 theorem ... := by 后面的 proof body 每行至少缩进 2 空格"""
    out = []
    for ln in block.splitlines():
        if ln.strip() == "":
            out.append("")
        elif ln.startswith("  "):
            out.append(ln)
        else:
            out.append("  " + ln)
    return "\n".join(out)

def safe_key(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(s))

def load_jsonl(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            items.append(json.loads(ln))
    return items

def reassemble_by_index(orig: str, fills_by_index: dict, fill_upto: int, missing_placeholder: str = HOLE) -> str:
    """
    orig: 原始多洞 sketch（含多个 <proof>）
    fills_by_index: {hole_index: fill}
    fill_upto: 只填前 fill_upto 个洞（与生成时 num_holes / MAX_HOLES_PER_PROOF 对齐）
    missing_placeholder: 缺洞时填什么。默认填回 <proof>（保证结构不乱、且 verify 会 fail）
    """
    parts = orig.split(HOLE)
    n = len(parts) - 1
    if n <= 0:
        return orig

    # 只填前 fill_upto 个洞（通常 fill_upto == min(n, num_holes)）
    m = max(0, min(fill_upto, n))

    out = []
    for i in range(n):
        out.append(parts[i])
        if i < m:
            out.append(fills_by_index.get(i, missing_placeholder))
        else:
            out.append(HOLE)  # 超出生成范围的洞保持不动
    out.append(parts[-1])
    return "".join(out)

def main():
    """
    ENV:
      IN_JSONL  = holes_all.jsonl（洞级输出；每行一洞）
      OUT_JSONL = generated_all.jsonl（题级输出；每行一题/一次attempt）
      STRICT_COMPLETE (optional): "1" -> 洞不齐就跳过该组
    """
    
    skipped_jsonl = os.environ.get("SKIPPED_JSONL", "").strip()
    skipped_fp = None
    if skipped_jsonl:
        Path(skipped_jsonl).parent.mkdir(parents=True, exist_ok=True)
        skipped_fp = open(skipped_jsonl, "w", encoding="utf-8")

    
    in_jsonl = os.environ["IN_JSONL"]
    out_jsonl = os.environ["OUT_JSONL"]
    strict_complete = bool(int(os.environ.get("STRICT_COMPLETE", "0")))

    records = load_jsonl(in_jsonl)

    # group by (rid, attempt)
    groups = defaultdict(list)
    for ex in records:
        rid = ex.get("rid") or ex.get("id") or ex.get("name") or ""
        attempt = int(ex.get("gen_attempt", 0))
        groups[(rid, attempt)].append(ex)

    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)

    assembled = 0
    skipped_incomplete = 0

    with open(out_jsonl, "w", encoding="utf-8") as w:
        for (rid, attempt), items in groups.items():
            # 以 hole_index 排序，保证确定性
            items.sort(key=lambda x: int(x.get("hole_index", 0)))

            base = dict(items[0])  # 继承原字段
            orig = base.get("orig_sketch") or base.get("sketch") or base.get("proof") or ""
            if not orig:
                continue

            # 期望洞数：优先 num_holes（与你生成时一致，能兼容 MAX_HOLES_PER_PROOF 截断）
            holes_expected = int(base.get("num_holes", orig.count(HOLE)))

            # 建立 hole_index -> record（若重复，保留第一个；也可改成挑更短/更长等策略）
            by_idx = {}
            duplicates = []
            for it in items:
                hi = int(it.get("hole_index", 0))
                if hi in by_idx:
                    duplicates.append(hi)
                    continue
                by_idx[hi] = it

            missing = [i for i in range(holes_expected) if i not in by_idx]
            holes_generated = len(by_idx)
            holes_complete = (len(missing) == 0) and (len(duplicates) == 0) and (holes_generated == holes_expected)

            if strict_complete and not holes_complete:
                skipped_incomplete += 1
                if skipped_fp:
                    skipped_fp.write(json.dumps({
                        "rid": rid,
                        "gen_attempt": attempt,
                        "holes_expected": holes_expected,
                        "holes_generated": holes_generated,
                        "missing_hole_indices": missing,
                        "duplicate_hole_indices": sorted(set(duplicates)),
                        "present_hole_indices": sorted(by_idx.keys()),
                        # 这两个有助于回溯原始数据
                        "id": base.get("id", None),
                        "name": base.get("name", None),
                    }, ensure_ascii=False) + "\n")
                continue


            # 回填用 fills_by_index
            fills_by_index = {hi: (by_idx[hi].get("gen_fill") or "") for hi in by_idx}

            # 组装回 orig（只填 holes_expected 个洞）
            cur = reassemble_by_index(orig, fills_by_index, fill_upto=holes_expected, missing_placeholder=HOLE)

            # 记录每个洞的 step（带各自 seed）
            gen_steps = []
            for hi in range(holes_expected):
                it = by_idx.get(hi)
                gen_steps.append({
                    "hole_index": hi,
                    "gen_fill": fills_by_index.get(hi, ""),
                    "gen_seed": (it.get("gen_seed") if it else None),
                    "gen_rank": (it.get("gen_rank") if it else None),
                    # 如果你觉得太大，可以删除下一行
                    "gen_raw":  (it.get("gen_raw")  if it else None),
                })

            # 拼接成完整 Lean
            header = base.get("header", "") or ""
            theorem = base.get("theorem", "") or ""
            final_body = normalize_body_for_by(cur)
            gen_final_lean = header.strip() + "\n\n" + theorem.rstrip() + "\n" + final_body + "\n"

            gen_uid = f"{rid}__a{attempt:02d}"
            base.update({
                "gen_uid": gen_uid,
                "gen_case_key": safe_key(gen_uid),

                "holes_expected": holes_expected,
                "holes_generated": holes_generated,
                "holes_complete": holes_complete,
                "missing_hole_indices": missing,
                "duplicate_hole_indices": sorted(set(duplicates)),

                "gen_num_fills": holes_expected,
                "gen_still_has_hole": (HOLE in cur),
                "gen_steps": gen_steps,
                "gen_final_lean": gen_final_lean,
            })

            w.write(json.dumps(base, ensure_ascii=False) + "\n")
            assembled += 1

    print(f"[assemble] groups_total={len(groups)} assembled={assembled} skipped_incomplete={skipped_incomplete}")
    print("[assemble] out ->", out_jsonl)
    
    if skipped_fp:
        skipped_fp.close()
        print("[assemble] skipped list ->", skipped_jsonl)


if __name__ == "__main__":
    main()
