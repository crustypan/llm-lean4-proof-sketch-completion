
# run_lean_verify.py
import os, json, time, subprocess, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


# tqdm (progress bar) — safe import
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x



def tail(s: str, n: int = 4000) -> str:
    if s is None:
        return ""
    return s[-n:]

def pick_code(ex: dict) -> str:
    # 优先取生成阶段的最终源码
    for k in ["gen_final_lean", "lean_code", "final_lean", "filled_lean", "lean_proof", "code"]:
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v

    # 兜底：尝试拼装
    header = ex.get("header", "") or ""
    theorem = ex.get("theorem", "") or ""
    proof = ex.get("proof_filled", "") or ex.get("proof", "") or ""
    return header + "\n\n" + theorem + "\n" + proof

def _safe_filename(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(s))

def run_one(ex: dict, lean_proj: str, out_dir: Path, timeout: int, lake_bin: str):
    rid = ex.get("id", "")
    case_key = ex.get("gen_case_key") or str(rid)
    case_key = _safe_filename(case_key)

    code = pick_code(ex)
    out_path = out_dir / f"{case_key}.lean"
    out_path.write_text(code, encoding="utf-8")

    t0 = time.time()
    cmd = [lake_bin, "env", "lean", str(out_path)]

    # 输出：保留输入全部字段，再追加验证字段
    base = dict(ex)

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
        tag = "pass" if ok else "lean_error"

        base.update({
            "pass": bool(ok),
            "tag": tag,
            "returncode": p.returncode,
            "time_sec": time.time() - t0,
            "lean_path": str(out_path),
            "stdout_tail": tail(p.stdout),
            "stderr_tail": tail(p.stderr),
        })
        return base

    except subprocess.TimeoutExpired as e:
        base.update({
            "pass": False,
            "tag": "timeout",
            "returncode": None,
            "time_sec": time.time() - t0,
            "lean_path": str(out_path),
            "stdout_tail": tail(getattr(e, "stdout", "")),
            "stderr_tail": tail(getattr(e, "stderr", "")),
        })
        return base

    except FileNotFoundError as e:
        base.update({
            "pass": False,
            "tag": "lake_not_found",
            "returncode": None,
            "time_sec": time.time() - t0,
            "lean_path": str(out_path),
            "stdout_tail": "",
            "stderr_tail": str(e),
        })
        return base

    except Exception as e:
        base.update({
            "pass": False,
            "tag": "other",
            "returncode": None,
            "time_sec": time.time() - t0,
            "lean_path": str(out_path),
            "stdout_tail": "",
            "stderr_tail": repr(e),
        })
        return base

def main():
    gen_jsonl = os.environ["GEN_JSONL"]
    lean_proj = os.environ["LEAN_PROJ"]
    out_jsonl = os.environ["OUT_JSONL"]
    timeout = int(os.environ.get("TIMEOUT", "1200"))
    jobs = int(os.environ.get("JOBS", "8"))
    lake_bin = os.environ.get("LAKE", "lake")

    out_jsonl_p = Path(out_jsonl)
    out_jsonl_p.parent.mkdir(parents=True, exist_ok=True)

    out_dir = out_jsonl_p.parent / "lean_files"
    out_dir.mkdir(parents=True, exist_ok=True)

    items = []
    with open(gen_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    t0 = time.time()
    passed = 0

    with open(out_jsonl, "w", encoding="utf-8") as w:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(run_one, it, lean_proj, out_dir, timeout, lake_bin) for it in items]
            
            
            pbar = tqdm(
                as_completed(futs),
                total=len(futs),
                desc="lean_verify",
                dynamic_ncols=True,
                mininterval = 10.0,   # ✅ 至少每 k 秒才刷新一次（默认 0.1s）
                maxinterval = 60.0,   # ✅ 最久 k 秒至少刷新一次（防止太久没动静）
                miniters = 50,        # ✅ 至少处理 k 个迭代才允许刷新（对 it/s 很快时有效）
                smoothing = 0.0,      # ✅ 可选：减少 ETA 抖动（不是降频，但更稳定）
            )
            
            done = 0
            for fut in pbar:
                r = fut.result()
                done += 1
                
                if r.get("pass"):
                    passed += 1
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
                
                # 显示已通过数量
                if (done % 20) == 0:
                    pbar.set_postfix_str(f"done = {done}   pass = {passed}")
                    

    print(f"verified {len(items)} files. pass={passed} ({passed/len(items)*100:.2f}%). elapsed={time.time()-t0:.1f}s")
    print("out:", out_jsonl)

if __name__ == "__main__":
    main()

