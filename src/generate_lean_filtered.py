
# run_stepwise_generate.py
import os, re, json, time, hashlib, subprocess
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

EOFILL = "<eofill>"
HOLE = "<proof>"

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


# -----------------------------
# helpers
# -----------------------------
def build_single_hole_sketch(orig: str, keep_idx: int, fills: dict, placeholder: str = "sorry") -> str:
    """
    orig: 原始多洞 proof body（不含 theorem header）
    keep_idx: 当前要生成的洞（0-based）
    fills: 已经“插回”的洞 {hole_index: fill_code}（是否使用由 IMMEDIATE_INSERT 控制）
    规则：
      - 已填洞：放 fill_code
      - 未填洞：keep_idx 位置放 <proof>，其余放 placeholder（默认 sorry）
    """
    parts = orig.split(HOLE)
    n = len(parts) - 1
    if n <= 0:
        return orig
    if keep_idx < 0 or keep_idx >= n:
        keep_idx = n - 1

    out = []
    for i in range(n):
        out.append(parts[i])
        if i in fills:
            out.append(fills[i])
        else:
            out.append(HOLE if i == keep_idx else placeholder)
    out.append(parts[-1])
    return "".join(out)


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


_CODEFENCE_RE = re.compile(r"^```.*?\n|\n```$", re.S)

def clean_fill(text: str) -> str:
    t = text.strip()
    t = _CODEFENCE_RE.sub("", t).strip()
    if EOFILL in t:
        t = t.split(EOFILL, 1)[0]
    t = t.replace(HOLE, "").strip("\n")
    return t.rstrip()


def _unbalanced(s: str) -> bool:
    return (s.count('[') > s.count(']')) or (s.count('(') > s.count(')')) or (s.count('{') > s.count('}'))

def _looks_truncated_lean(fill: str) -> bool:
    t = fill.rstrip()
    if t.endswith('[') or t.endswith(',') or t.endswith(':=') or t.endswith('by') or t.endswith('rw') or t.endswith('simp'):
        return True
    if 'rw [' in t and t.count('[') > t.count(']'):
        return True
    if 'linarith [' in t and t.count('[') > t.count(']'):
        return True
    return _unbalanced(t)


def _stable_int_seed(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % (2**31 - 1)

def _safe_key(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(s))

def tail(s: str, n: int = 2000) -> str:
    if s is None:
        return ""
    return s[-n:]


SYSTEM_PROMPT = (
    "You are a Lean 4 proof completion model.\n"
    "Given a Lean 4 sketch that contains exactly ONE <proof> hole, you must output ONLY the Lean code "
    "that should replace the <proof>.\n"
    "Ignore any 'sorry' in the input; do not try to replace or modify it.\n"
    "Do not output explanations. Do not repeat the input.\n"
    "Do NOT include the token 'sorry' anywhere in your output.\n"
    "End your output with the token <eofill>.\n"
)

def build_prompt(tokenizer, header: str, theorem: str, sketch_state: str) -> str:
    user = (
        "Replace the <proof> in the following Lean 4 sketch.\n"
        "Ignore any occurrences of 'sorry'.\n\n"
        "Do NOT include the token 'sorry' anywhere in your output.\n"
        "[HEADER]\n"
        f"{header}\n\n"
        "[THEOREM+SKETCH]\n"
        f"{theorem}\n"
        f"{sketch_state}\n"
    )
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"<<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n<<USER>>\n{user}\n<</USER>>\n\n<<ASSISTANT>>\n"


def resolve_adapter_dir(adapter_dir: str) -> str:
    p = Path(adapter_dir)
    if (p / "adapter_config.json").exists():
        return str(p)
    cands = []
    for ck in p.glob("checkpoint-*"):
        if (ck / "adapter_config.json").exists():
            m = re.search(r"checkpoint-(\d+)", ck.name)
            step = int(m.group(1)) if m else -1
            cands.append((step, ck))
    if cands:
        cands.sort()
        return str(cands[-1][1])
    raise FileNotFoundError(f"adapter_config.json not found under: {p}")


def load_jsonl(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            items.append(json.loads(ln))
    return items


def compile_with_lean(code: str, lean_proj: str, lake_bin: str, tmp_dir: Path, case_key: str, timeout: int):
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fp = tmp_dir / f"{case_key}.lean"
    fp.write_text(code, encoding="utf-8")

    cmd = [lake_bin, "env", "lean", str(fp)]
    t0 = time.time()
    try:
        p = subprocess.run(
            cmd,
            cwd=lean_proj,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=os.environ.copy(),
        )
        ok = (p.returncode == 0)
        return ok, p.returncode, time.time() - t0, tail(p.stdout), tail(p.stderr), str(fp)
    except subprocess.TimeoutExpired as e:
        return False, None, time.time() - t0, tail(getattr(e, "stdout", "")), tail(getattr(e, "stderr", "")), str(fp)
    except Exception as e:
        return False, None, time.time() - t0, "", repr(e), str(fp)


def main():
    cfg_path    = os.environ["RUN_CONFIG_PATH"]
    sketch_jsonl = os.environ["SKETCH_JSONL"]
    out_jsonl    = os.environ["OUT_JSONL"]

    max_new   = int(os.environ.get("MAX_NEW_TOKENS_PER_HOLE", "256"))
    max_holes = int(os.environ.get("MAX_HOLES_PER_PROOF", "32"))

    # 这里的 K_ATTEMPTS 解释为：每个洞一次性生成的候选数量 k
    k_per_hole = int(os.environ.get("K_PER_HOLE", os.environ.get("K_ATTEMPTS", "1")))
    seed_base  = int(os.environ.get("GEN_SEED_BASE", "1234"))

    # 外层“整题重复次数”（默认 1）。需要更多可在 torchrun env 里传 N_ATTEMPTS
    n_attempts = int(os.environ.get("N_ATTEMPTS", "1"))

    do_sample  = bool(int(os.environ.get("DO_SAMPLE", "0")))
    temperature = float(os.environ.get("TEMPERATURE", "0.7"))
    top_p       = float(os.environ.get("TOP_P", "0.95"))

    immediate_insert = bool(int(os.environ.get("IMMEDIATE_INSERT", "0")))

    # Lean 编译器（洞级筛选）
    lean_proj = os.environ.get("LEAN_PROJ", "").strip()
    lake_bin  = os.environ.get("LAKE", "lake").strip()
    hole_timeout = int(os.environ.get("HOLE_VERIFY_TIMEOUT", "120"))
    keep_pass_max = int(os.environ.get("KEEP_PASSING_MAX", "8"))  # 每洞最多保留多少个“可编译候选”用于记录
    gen_batch_k = int(os.environ.get("GEN_BATCH_K", str(min(k_per_hole, 8))))  # 防止 num_return_sequences 太大 OOM

    if not lean_proj:
        raise RuntimeError("LEAN_PROJ is required for per-hole compilation filtering. Please pass LEAN_PROJ in env.")
    if not Path(lean_proj).exists():
        raise RuntimeError(f"LEAN_PROJ not found: {lean_proj}")

    adapter_dir   = os.environ.get("ADAPTER_DIR", "")
    tokenizer_dir = os.environ.get("TOKENIZER_DIR", "")

    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    base_model = cfg["base_model"]
    out_dir_root = cfg.get("out_dir", "")

    if not adapter_dir:
        adapter_dir = out_dir_root
    adapter_dir = resolve_adapter_dir(adapter_dir)

    if not tokenizer_dir:
        tokenizer_dir = out_dir_root

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    outp = Path(out_jsonl)
    if outp.suffix == ".jsonl":
        out_file = outp
        out_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        outp.mkdir(parents=True, exist_ok=True)
        out_file = outp / f"gen_rank{rank}.jsonl"

    if rank == 0:
        print("base_model:", base_model)
        print("tokenizer_dir:", tokenizer_dir)
        print("adapter_dir:", adapter_dir)
        print("out_file:", out_file)
        print("k_per_hole:", k_per_hole, "n_attempts:", n_attempts, "immediate_insert:", immediate_insert)
        print("lean_proj:", lean_proj, "lake_bin:", lake_bin, "hole_timeout:", hole_timeout)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [HOLE, EOFILL]})
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"

    eofill_id = tokenizer.convert_tokens_to_ids(EOFILL)
    stop_id = eofill_id if (eofill_id is not None and eofill_id >= 0) else tokenizer.eos_token_id

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_cfg,
        device_map={"": local_rank},
        torch_dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    model.eval()

    data = load_jsonl(sketch_jsonl)
    if any(("split" in x) for x in data):
        test = [x for x in data if x.get("split") == "test"] or \
               [x for x in data if x.get("split") == "valid"] or data
    else:
        test = data

    # ---- tasks: (ex, attempt) ----
    tasks = []
    for ex in test:
        rid = ex.get("id", "")
        orig = ex.get("sketch", "") or ex.get("proof", "") or ""
        num_holes = orig.count(HOLE)
        if num_holes <= 0:
            continue
        num_holes = min(num_holes, max_holes)
        for attempt in range(n_attempts):
            tasks.append({"ex": ex, "rid": rid, "attempt": attempt, "num_holes": num_holes, "orig": orig})

    total_groups_global = len(tasks)
    tasks = tasks[rank::world]
    total_groups_local = len(tasks)
    total_holes_local = sum(t["num_holes"] for t in tasks)

    if rank == 0:
        print(f"[gen] groups_global={total_groups_global} world={world}")
    print(f"[gen] rank={rank} groups_local={total_groups_local} holes_local={total_holes_local}")

    tmp_dir = out_file.parent / f"tmp_lean_rank{rank}"
    t0 = time.time()
    n_lines = 0

    with open(out_file, "w", encoding="utf-8") as w:
        pbar = tqdm(
            total=total_holes_local,
            desc=f"gen rank{rank}",
            dynamic_ncols=True,
            disable=(rank != 0),
            mininterval=float(os.environ.get("TQDM_MININTERVAL", "60")),
            maxinterval=float(os.environ.get("TQDM_MAXINTERVAL", "180")),
            miniters=int(os.environ.get("TQDM_MINITERS", "1")),
            smoothing=float(os.environ.get("TQDM_SMOOTHING", "0.0")),
        )

        for t in tasks:
            ex = t["ex"]
            rid = t["rid"]
            attempt = t["attempt"]
            num_holes = t["num_holes"]
            orig = t["orig"]

            header = ex.get("header", "") or ""
            theorem = ex.get("theorem", "") or ""

            # 这个 dict 永远记录“最终选中”的 fill（用于最终 assemble）
            fills_selected = {}

            # 这个 dict 决定“后续洞生成时是否能看到前面的 fill”
            fills_visible = {}  # immediate_insert=1 时会同步为 fills_selected

            failed = False

            for hi in range(num_holes):
                # 当前洞的输入 sketch（其余洞 sorry；是否含前面洞的 fill 由 IMMEDIATE_INSERT 决定）
                if immediate_insert:
                    fills_visible = dict(fills_selected)
                else:
                    fills_visible = {}

                single = build_single_hole_sketch(orig, hi, fills_visible, placeholder="sorry")

                prompt = build_prompt(tokenizer, header, theorem, single)
                enc = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=int(cfg.get("max_len", 8192)),
                )
                input_ids = enc["input_ids"].to(model.device)
                attn = enc["attention_mask"].to(model.device)
                in_len = input_ids.shape[-1]

                # 多候选时必须采样
                do_sample_eff = do_sample or (k_per_hole > 1)

                # 生成 k 个候选（分批避免 OOM）
                seed = (seed_base + _stable_int_seed(f"{rid}__a{attempt}__h{hi}")) % (2**31 - 1)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

                candidates = []  # list of (fill, raw)
                remaining = k_per_hole
                while remaining > 0:
                    b = min(gen_batch_k, remaining)
                    gen_kwargs = dict(
                        input_ids=input_ids,
                        attention_mask=attn,
                        max_new_tokens=max_new,
                        do_sample=do_sample_eff,
                        eos_token_id=stop_id,
                        pad_token_id=tokenizer.pad_token_id,
                        num_return_sequences=b,
                    )
                    if do_sample_eff:
                        gen_kwargs.update(dict(temperature=temperature, top_p=top_p))

                    with torch.inference_mode():
                        outs = model.generate(**gen_kwargs)

                    # outs: [b, in_len + ...]
                    for seq in outs:
                        raw = tokenizer.decode(seq[in_len:], skip_special_tokens=False)
                        fill = clean_fill(raw)
                        candidates.append((fill, raw))

                    remaining -= b

                # 对每个候选：插回当前洞 -> Lean 编译筛选
                passing = []  # list of dict {fill, raw, compile_*}
                tested = 0

                for ci, (fill, raw) in enumerate(candidates):
                    tested += 1

                    # 允许用 truncated 直接交给 lean 试；也可以先过滤：
                    # if _looks_truncated_lean(fill): continue

                    # 构造用于编译的 proof body：single 里只有一个 <proof>
                    body = single.replace(HOLE, fill, 1)
                    lean_body = normalize_body_for_by(body)
                    full_code = header.strip() + "\n\n" + theorem.rstrip() + "\n" + lean_body + "\n"

                    case_key = _safe_key(f"{rid}__a{attempt:02d}__h{hi:03d}")
                    ok, rc, tsec, out_tail, err_tail, lean_path = compile_with_lean(
                        full_code, lean_proj, lake_bin, tmp_dir, case_key, hole_timeout
                    )

                    if ok:
                        passing.append({
                            "cand_index": ci,
                            "fill": fill,
                            "raw": raw,
                            "compile_returncode": rc,
                            "compile_time_sec": tsec,
                        })
                        if len(passing) >= keep_pass_max:
                            break  # 控制记录规模/时间；需要全量可把 KEEP_PASSING_MAX=k_per_hole

                if len(passing) == 0:
                    # 该洞无任何可编译候选 -> 该题失败，跳下一题
                    failed = True

                    rec = dict(ex)
                    rec.update({
                        "rid": rid,
                        "gen_attempt": attempt,
                        "hole_index": hi,
                        "num_holes": num_holes,
                        "gen_seed": seed,
                        "gen_rank": rank,
                        "gen_world": world,

                        "immediate_insert": bool(immediate_insert),

                        "orig_sketch": orig,
                        "single_sketch": single,

                        "k_per_hole": k_per_hole,
                        "tested_candidates": tested,
                        "passing_candidates": 0,
                        "fail_reason": "no_compilable_candidate",

                        "gen_fill": "",
                        "gen_raw": "",
                    })

                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_lines += 1
                    pbar.update(1)
                    break

                # 选择一个通过者作为“本洞最终填充”
                chosen = passing[0]  # 默认：第一个通过者（最稳定、可复现）
                chosen_fill = chosen["fill"]
                chosen_raw  = chosen["raw"]

                fills_selected[hi] = chosen_fill
                if immediate_insert:
                    fills_visible[hi] = chosen_fill

                rec = dict(ex)
                rec.update({
                    "rid": rid,
                    "gen_attempt": attempt,
                    "hole_index": hi,
                    "num_holes": num_holes,
                    "gen_seed": seed,
                    "gen_rank": rank,
                    "gen_world": world,

                    "immediate_insert": bool(immediate_insert),

                    "orig_sketch": orig,
                    "single_sketch": single,

                    "k_per_hole": k_per_hole,
                    "tested_candidates": tested,
                    "passing_candidates": len(passing),
                    "chosen_cand_index": chosen["cand_index"],

                    # 记录通过候选（截断到 keep_pass_max）
                    "pass_fills": [p["fill"] for p in passing],

                    # assemble 仍然读这个字段
                    "gen_fill": chosen_fill,
                    "gen_raw": chosen_raw,
                })

                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_lines += 1
                pbar.update(1)

                if (n_lines % 1000) == 0:
                    pbar.set_postfix_str(f"written={n_lines}")

            # 若失败则该题直接结束（不生成后续洞）
            if failed:
                continue

        pbar.close()

    if rank == 0:
        print(f"done. rank0 wrote {n_lines} lines. elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
