import json, os, subprocess, sys, time, urllib.request

CTRL = r"F:\Strata\control"
PY = os.path.join(CTRL, "python", "python.exe")
BASE = json.load(open(r"F:\Strata\install\strata-iq3_s.json", encoding="utf-8"))
BENCH = r"F:\tmp-strata-bench"
PORT = 8182

ENV = dict(os.environ, STRATA_PROFILE_DIR=os.path.join(BENCH, "profiles"),
           STRATA_RUNTIME_DIR=os.path.join(BENCH, "runtime"),
           STRATA_COMPONENTS_DIR=r"F:\Strata\components")

FILLER = ("The quick brown fox jumps over the lazy dog while the engineer measures memory bandwidth, "
          "cache behaviour and the cost of every hand-off between two graphics cards. ") * 200


def profile(name, env_extra=None, args_extra=None):
    cfg = json.loads(json.dumps(BASE))
    cfg["port"] = PORT
    cfg["api_key"] = "none"
    cfg["log"] = os.path.join(BENCH, name + ".log")
    cfg["drafting"] = "dflash2"
    if args_extra and "--gpu" in args_extra:
        cfg["gpu"] = 0
        cfg["gpu_roles"] = {"0": "main", "1": "off"}
    if args_extra:
        args = list(cfg.get("args") or [])
        for i, a in enumerate(list(args)):
            if a == "--pool-workers" and "--pool-workers" in args_extra:
                args[i + 1] = args_extra[args_extra.index("--pool-workers") + 1]
        extra = list(args_extra)
        if "--pool-workers" in args_extra:
            j = extra.index("--pool-workers")
            del extra[j:j + 2]
        cfg["args"] = args + extra
    cfg["env"] = {**(cfg.get("env") or {}), **(env_extra or {})}
    path = os.path.join(BENCH, name + ".json")
    json.dump(cfg, open(path, "w", encoding="utf-8"), indent=2)
    d = os.path.join(BENCH, "profiles")
    os.makedirs(d, exist_ok=True)
    json.dump({"id": name, "overrides": {"port": PORT}, "schema_version": 1, "source_config": path},
              open(os.path.join(d, name + ".json"), "w", encoding="utf-8"), indent=2)
    return path


def run(*a, timeout=3600):
    p = subprocess.run([PY, "strata.py", *a], cwd=CTRL, capture_output=True, text=True, timeout=timeout, env=ENV)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def ask(text, max_tokens=4):
    body = json.dumps({"model": "bench", "messages": [{"role": "user", "content": text}],
                       "temperature": 0, "max_tokens": max_tokens, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read().decode())


def bench(name, env_extra, note, args_extra=None):
    print(f"\n=== {name}: {note}", flush=True)
    p = profile(name, env_extra, args_extra)
    t0 = time.time()
    rc, out = run("runtime", "start", "--profile", name, "--wait", "900", "--output", "json")
    if rc != 0:
        print("start failed", out[-800:], flush=True)
        return {"name": name, "error": out[-400:]}
    print(f"started in {time.time()-t0:.0f}s", flush=True)
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            break
        except Exception:
            time.sleep(2)
    rows = []
    try:
        ask("hello", 2)
        for i in range(3):
            r = ask(FILLER + f"\n\nRun {i}. Reply with one word.", 4)
            tm = r.get("timings", {})
            rows.append((r.get("usage", {}).get("prompt_tokens"), tm.get("prompt_per_second")))
            print(f"  prompt {rows[-1][0]} tokens, prefill {rows[-1][1]:.1f} tok/s", flush=True)
    except Exception as e:
        print("request failed:", e, flush=True)
    run("runtime", "stop", "--profile", name, "--output", "json", timeout=600)
    vals = sorted(x[1] for x in rows if x[1])
    return {"name": name, "note": note, "tokens": rows[0][0] if rows else None,
            "prefill_med": round(vals[len(vals)//2], 1) if vals else None,
            "prefill_all": [round(v, 1) for v in vals]}


if __name__ == "__main__":
    arms = {
        "p0": (None, "baseline"),
        "p2": ({"STRATA_DENSE_F32": "1"}, "dense GEMM as cublasSgemm FP32"),
        "p3": ({"STRATA_PREFILL_TIMING": "1"}, "phase timing on"),
        "p4": (None, "--prefill-pipe 1000"),
        "p5": (None, "--prefill-pipe 3000"),
        "p6": (None, "--prefill-pipe 6000"),
        "p7": ({"STRATA_PREFILL_TIMING": "1"}, "pipe 1000 + timing"),
        "p8": ({"STRATA_DENSE_F32": "1"}, "DENSE_F32 + 20 pool workers"),
        "p9": ({"STRATA_DENSE_F32": "1"}, "DENSE_F32 + 20 pool workers + pipe 3000"),
        "p10": ({"STRATA_DENSE_F32": "1"}, "DENSE_F32 + pcie-frac 0.6"),
        "p11": ({"STRATA_DENSE_F32": "1"}, "ONE card (GPU0), DENSE_F32"),
        "p12": (None, "ONE card (GPU0), defaults"),
        "p13": ({"STRATA_DENSE_F32": "1"}, "ONE card, DENSE_F32, pcie-frac 0"),
    }
    want = sys.argv[1:] or ["p2"]
    pipes = {"p4": ["--prefill-pipe", "1000"], "p5": ["--prefill-pipe", "3000"], "p6": ["--prefill-pipe", "6000"],
             "p7": ["--prefill-pipe", "1000"],
             "p8": ["--pool-workers", "20"],
             "p9": ["--pool-workers", "20", "--prefill-pipe", "3000"],
             "p10": ["--pcie-frac", "0.6"],
             "p11": ["--gpu", "0"],
             "p12": ["--gpu", "0"],
             "p13": ["--gpu", "0", "--pcie-frac", "0"]}
    res = [bench(k, arms[k][0], arms[k][1], pipes.get(k)) for k in want]
    print("\n== summary")
    for r in res:
        print(json.dumps(r, ensure_ascii=False))
    json.dump(res, open(os.path.join(BENCH, "prefill.json"), "w", encoding="utf-8"), indent=2)
