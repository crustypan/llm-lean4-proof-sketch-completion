
# run_stepwise_generate.py
import os, re, json, time, hashlib
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

EOFILL = "<eofill>"
HOLE = "<proof>"


# tqdm (progress bar) — safe import
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x



def count_holes(sketch: str) -> int:
    return sketch.count(HOLE)




def build_single_hole_sketch(orig: str, keep_idx: int, fills: dict, placeholder: str = "sorry") -> str:
    """
    orig: 原始多洞 sketch（只含 proof body，不含 theorem header）
    keep_idx: 要保留的洞（按 orig 中 <proof> 出现顺序，0-based）
    fills: 已经插入的洞 {hole_index: fill_code}；未插入的洞不在 dict 中
    placeholder: 其它未填洞用什么替换（默认 sorry）
    规则：已填洞保留 fill_code；未填洞中只有 keep_idx 保留 <proof>，其它换成 placeholder。
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





SYSTEM_PROMPT = (
    "You are a Lean 4 proof completion model.\n"
    "Given a Lean 4 sketch that contains exactly ONE <proof> hole, you must output ONLY the Lean code "
    "that should replace the <proof>.\n"
    "Ignore any 'sorry' in the input; do not try to replace or modify it.\n"
    "Do not output explanations. Do not repeat the input.\n"
    "Do NOT include the token 'sorry' anywhere in your output.\n"
    "End your output with the token <eofill>.\n"
)

def resolve_adapter_dir(adapter_dir: str) -> str:
    """adapter_dir 可以是 checkpoint-xxx 或 out_dir 根目录（自动挑最新 checkpoint）"""
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

def find_first_hole_linecol(s: str):
    idx = s.find(HOLE)
    if idx < 0:
        return -1, -1
    line = s.count("\n", 0, idx)
    col = idx - (s.rfind("\n", 0, idx) + 1)
    return line, col



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

def _stable_int_seed(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % (2**31 - 1)

def _safe_key(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(s))

def main():
    cfg_path = os.environ["RUN_CONFIG_PATH"]
    sketch_jsonl = os.environ["SKETCH_JSONL"]
    out_jsonl = os.environ["OUT_JSONL"]

    max_new = int(os.environ.get("MAX_NEW_TOKENS_PER_HOLE", "256"))
    max_holes = int(os.environ.get("MAX_HOLES_PER_PROOF", "32"))
    do_sample = bool(int(os.environ.get("DO_SAMPLE", "0")))
    temperature = float(os.environ.get("TEMPERATURE", "0.2"))
    top_p = float(os.environ.get("TOP_P", "0.95"))

    # pass@k：每题生成几次完整尝试
    k_attempts = int(os.environ.get("K_ATTEMPTS", "1"))
    seed_base  = int(os.environ.get("GEN_SEED_BASE", "1234"))

    adapter_dir = os.environ.get("ADAPTER_DIR", "")
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
    # DDP 推荐传目录：out_jsonl=/path/to/gen_dir
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
        print("K_ATTEMPTS:", k_attempts)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [HOLE, EOFILL]})
    tokenizer.truncation_side = "right"
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

    # 若存在 split 字段则优先按 split==test；否则默认整文件就是测试集
    if any(("split" in x) for x in data):
        test = [x for x in data if x.get("split") == "test"]
        if not test:
            test = [x for x in data if x.get("split") == "valid"]
        if not test:
            test = data
    else:
        test = data



    t0 = time.time()
    n_lines = 0
    
    
    # =========  展开成“洞级任务”列表，然后按 rank 分片 =========
    immediate_insert = bool(int(os.environ.get("IMMEDIATE_INSERT", "0")))

    # =========  展开成“(题, attempt) 级任务”，rank 之间分题并行 =========
    tasks = []
    for ex in test:
        rid = ex.get("id", "")
        orig = ex.get("sketch", "") or ex.get("proof", "") or ""
        num_holes = orig.count(HOLE)
        if num_holes <= 0:
            continue
        num_holes = min(num_holes, max_holes)

        for attempt in range(k_attempts):
            tasks.append({
                "ex": ex,
                "rid": rid,
                "attempt": attempt,
                "num_holes": num_holes,
                "orig": orig,
            })

    total_groups_global = len(tasks)
    tasks = tasks[rank::world]
    total_groups_local = len(tasks)

    # 进度条按“洞”更直观：每个 group 有 num_holes 个洞
    total_holes_local = sum(t["num_holes"] for t in tasks)

    if rank == 0:
        print(f"[gen] groups_global={total_groups_global} world={world} immediate_insert={immediate_insert}")
    print(f"[gen] rank={rank} groups_local={total_groups_local} holes_local={total_holes_local}")


    t0 = time.time()
    n_lines = 0


    with open(out_file, "w", encoding="utf-8") as w:

        pbar = tqdm(
            total=total_holes_local,
            desc=f"gen rank{rank}",
            dynamic_ncols=True,
            disable=(rank != 0),   # Notebook 建议只显示 rank0，避免刷屏
            mininterval = float(os.environ.get("TQDM_MININTERVAL", "60")),      # ✅ 至少每 k 秒才刷新一次（默认 0.1s）
            maxinterval = float(os.environ.get("TQDM_MAXINTERVAL", "180")),     # ✅ 最久 k 秒至少刷新一次（防止太久没动静）
            miniters = int(os.environ.get("TQDM_MINITERS", "1")),               # 至少处理 k 个迭代才允许刷新（对 it/s 很快时有效）
            smoothing = float(os.environ.get("TQDM_SMOOTHING", "0.0")),         # ✅ 可选：减少 ETA 抖动（不是降频，但更稳定）
        )

        n_lines = 0

        for t in tasks:
            ex = t["ex"]
            rid = t["rid"]
            attempt = t["attempt"]
            num_holes = t["num_holes"]
            orig = t["orig"]

            header = ex.get("header", "") or ""
            theorem = ex.get("theorem", "") or ""

            # immediate_insert=1 时：fills_so_far 会逐步增长，后续洞可见前面填好的代码
            # immediate_insert=0 时：fills_so_far 固定为空，所有洞生成互不可见
            fills_so_far = {}

            for hi in range(num_holes):

                # 关键：构造“单洞可见”的输入 sketch
                if immediate_insert:
                    single = build_single_hole_sketch(orig, hi, fills_so_far, placeholder="sorry")
                else:
                    single = build_single_hole_sketch(orig, hi, {}, placeholder="sorry")

                seed = (seed_base + _stable_int_seed(f"{rid}__a{attempt}__h{hi}")) % (2**31 - 1)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

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

                do_sample_eff = do_sample or (k_attempts > 1)

                with torch.no_grad():
                    gen = model.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        max_new_tokens=max_new,
                        do_sample=do_sample_eff,
                        temperature=temperature if do_sample_eff else None,
                        top_p=top_p if do_sample_eff else None,
                        eos_token_id=stop_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                out_text = tokenizer.decode(gen[0][in_len:], skip_special_tokens=False)
                fill = clean_fill(out_text)

                # 截断/括号不平衡：重试一次（保留你原逻辑）
                if _looks_truncated_lean(fill):
                    gen2 = model.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        max_new_tokens=max_new * 2,
                        do_sample=True,
                        temperature=max(0.7, temperature),
                        top_p=top_p,
                        eos_token_id=stop_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    out_text2 = tokenizer.decode(gen2[0][in_len:], skip_special_tokens=False)
                    fill2 = clean_fill(out_text2)
                    if not _looks_truncated_lean(fill2):
                        out_text, fill = out_text2, fill2

                # ✅ 是否立即插回
                if immediate_insert:
                    fills_so_far[hi] = fill

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
                    "single_sketch": single,   # 该洞生成时实际看到的 sketch（含/不含已插入代码）

                    "gen_fill": fill,
                    "gen_raw": out_text,
                })

                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_lines += 1
                pbar.update(1)

                if (n_lines % 1000) == 0:
                    pbar.set_postfix_str(f"written={n_lines}")

        pbar.close()

    if rank == 0:
        print(f"done. rank0 wrote {n_lines} HOLE-lines. elapsed={time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
