
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

def keep_only_one_hole(sketch: str, keep_idx: int, placeholder: str = "sorry") -> str:
    """
    把除了 keep_idx 以外的 <proof> 全部替换为 placeholder（默认 sorry），只保留一个 <proof>
    keep_idx: 0-based，按从上到下出现顺序计数
    """
    parts = sketch.split(HOLE)
    n = len(parts) - 1
    if n <= 0:
        return sketch
    if keep_idx < 0 or keep_idx >= n:
        keep_idx = n - 1

    out = []
    for i in range(n):
        out.append(parts[i])
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
    tasks = []
    for ex in test:
        rid = ex.get("id", "")
        header = ex.get("header", "") or ""
        theorem = ex.get("theorem", "") or ""

        orig = ex.get("sketch", "")
        if not orig:
            orig = ex.get("proof", "") or ""

        num_holes = orig.count(HOLE)
        if num_holes <= 0:
            continue
        num_holes = min(num_holes, max_holes)

        for attempt in range(k_attempts):
            for hi in range(num_holes):
                single = keep_only_one_hole(orig, hi, placeholder="sorry")
                seed = (seed_base + _stable_int_seed(f"{rid}__a{attempt}__h{hi}")) % (2**31 - 1)
                tasks.append({
                    "ex": ex,
                    "rid": rid,
                    "attempt": attempt,
                    "hole_index": hi,          # 0-based
                    "num_holes": num_holes,
                    "orig": orig,
                    "single": single,
                    "seed": seed,
                })


    
    # 洞级并行分片（关键！不再是“题级分片”）
    total_tasks_global = len(tasks)  # 切分前的总任务数（全 rank 总和）
    tasks = tasks[rank::world]
    total_tasks_local = len(tasks)
    
    
    if rank == 0:
        print(f"[gen] global_tasks={total_tasks_global}  world={world}")
    print(f"[gen] rank={rank} local_tasks={total_tasks_local}")

    t0 = time.time()
    n_lines = 0
    with open(out_file, "w", encoding="utf-8") as w:
    
        # 每个 rank 一个进度条；position=rank 便于多条同时显示（终端更友好）
        pbar = tqdm(
            tasks,
            total=total_tasks_local,
            desc=f"gen rank{rank}",
            dynamic_ncols=True,
            position=rank,
            leave=(rank == 0),
            mininterval = 60.0,   # ✅ 至少每 k 秒才刷新一次（默认 0.1s）
            maxinterval = 180.0,   # ✅ 最久 k 秒至少刷新一次（防止太久没动静）
            miniters = 50,        # ✅ 至少处理 k 个迭代才允许刷新（对 it/s 很快时有效）
            smoothing = 0.0,      # ✅ 可选：减少 ETA 抖动（不是降频，但更稳定）

        )
        
        for t in pbar:
            ex = t["ex"]
            rid = t["rid"]
            attempt = t["attempt"]
            hi = t["hole_index"]

            header = ex.get("header", "") or ""
            theorem = ex.get("theorem", "") or ""

            torch.manual_seed(t["seed"])
            torch.cuda.manual_seed_all(t["seed"])

            prompt = build_prompt(tokenizer, header, theorem, t["single"])
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
                    
                    

            rec = dict(ex)
            rec.update({
                # ---- 单洞记录（之后 assemble 用这些字段组装）----
                "rid": rid,
                "gen_attempt": attempt,
                "hole_index": hi,
                "num_holes": t["num_holes"],
                "gen_seed": t["seed"],
                "gen_rank": rank,
                "gen_world": world,

                "orig_sketch": t["orig"],
                "single_sketch": t["single"],

                "gen_fill": fill,
                "gen_raw": out_text,
            })

            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_lines += 1
            
            
            # 降低刷新频率，避免开销
            if (n_lines % 1000) == 0:
                pbar.set_postfix_str(f"written={n_lines}")

    if rank == 0:
        print(f"done. rank0 wrote {n_lines} HOLE-lines. elapsed={time.time()-t0:.1f}s")




if __name__ == "__main__":
    main()
