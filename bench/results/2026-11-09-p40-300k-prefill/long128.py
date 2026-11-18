import json, os, subprocess, sys, time, urllib.request

CTRL = r"F:\Strata\control"
PY = os.path.join(CTRL, "python", "python.exe")
BASE = json.load(open(r"F:\Strata\install\strata-iq3_s.json", encoding="utf-8"))
BENCH = r"F:\tmp-strata-bench"
PORT = 8184

ENV = dict(os.environ, STRATA_PROFILE_DIR=os.path.join(BENCH, "profiles"),
           STRATA_RUNTIME_DIR=os.path.join(BENCH, "runtime"),
           STRATA_COMPONENTS_DIR=r"F:\Strata\components")

UNIT = ("The quick brown fox jumps over the lazy dog while the engineer measures memory bandwidth, "
        "cache behaviour and the cost of every hand-off between two graphics cards. ")
TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 127000
PROMPT = UNIT * max(1, TARGET // 30)


EXTRA = {}
if len(sys.argv) > 3:
    for kv in sys.argv[3].split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            EXTRA[k.strip()] = v.strip()


def profile(name, one_card=False, dense=True, timing=False):
    cfg = json.loads(json.dumps(BASE))
    cfg["port"] = PORT
    cfg["api_key"] = "none"
    cfg["log"] = os.path.join(BENCH, name + ".log")
    cfg["drafting"] = "dflash2"
    env = {**(cfg.get("env") or {})}
    if dense:
        env["STRATA_DENSE_F32"] = "1"
    else:
        env.pop("STRATA_DENSE_F32", None)
    if timing:
        env["STRATA_PREFILL_TIMING"] = "1"
    env.update(EXTRA)
    cfg["env"] = env
    if len(sys.argv) > 4 and sys.argv[4]:
        cfg["args"] = list(cfg.get("args") or []) + sys.argv[4].split(",")
    if len(sys.argv) > 5 and sys.argv[5]:
        # rewrite a flag that is already in the config (--pool-workers N) instead of adding a second one
        flag, val = sys.argv[5].split("=", 1)
        a = list(cfg.get("args") or [])
        for i, x in enumerate(a):
            if x == flag and i + 1 < len(a):
                a[i + 1] = val
                break
        else:
            a += [flag, val]
        cfg["args"] = a
    if one_card:
        cfg["args"] = list(cfg.get("args") or []) + ["--gpu", "0"]
        cfg["gpu"] = 0
        cfg["gpu_roles"] = {"0": "main", "1": "off"}
    path = os.path.join(BENCH, name + ".json")
    json.dump(cfg, open(path, "w", encoding="utf-8"), indent=2)
    d = os.path.join(BENCH, "profiles")
    os.makedirs(d, exist_ok=True)
    json.dump({"id": name, "overrides": {"port": PORT}, "schema_version": 1, "source_config": path},
              open(os.path.join(d, name + ".json"), "w", encoding="utf-8"), indent=2)


def run(*a, timeout=3600):
    p = subprocess.run([PY, "strata.py", *a], cwd=CTRL, capture_output=True, text=True, timeout=timeout, env=ENV)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def ask(text, max_tokens):
    body = json.dumps({"model": "bench", "messages": [{"role": "user", "content": text}],
                       "temperature": 0, "max_tokens": max_tokens, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read().decode())


def bench(name, one_card, dense, timing, note):
    print(f"\n=== {name}: {note}", flush=True)
    profile(name, one_card, dense, timing)
    t0 = time.time()
    rc, out = run("runtime", "start", "--profile", name, "--wait", "900", "--output", "json")
    if rc != 0:
        print("start failed", out[-500:], flush=True)
        return {"name": name, "error": out[-300:]}
    print(f"started in {time.time()-t0:.0f}s", flush=True)
    import threading
    stop = {"v": False}

    def sample():
        with open(os.path.join(BENCH, name + ".temps"), "w") as f:
            while not stop["v"]:
                o = subprocess.run(["nvidia-smi", "--query-gpu=index,temperature.gpu,clocks.sm,power.draw",
                                    "--format=csv,noheader"], capture_output=True, text=True).stdout
                f.write(f"{time.time():.0f} " + o.replace("\n", " | ") + "\n")
                f.flush()
                time.sleep(5)

    threading.Thread(target=sample, daemon=True).start()
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            break
        except Exception:
            time.sleep(2)
    rec = {"name": name, "note": note}
    try:
        t1 = time.time()
        r = ask(PROMPT, 4)
        tm = r.get("timings", {})
        rec.update({"prompt_tokens": r.get("usage", {}).get("prompt_tokens"),
                    "prefill_tps": round(tm.get("prompt_per_second") or 0, 1),
                    "wall_s": round(time.time() - t1, 1)})
        print(json.dumps(rec, ensure_ascii=False), flush=True)
    except Exception as e:
        print("request failed:", e, flush=True)
        rec["error"] = str(e)
    run("runtime", "stop", "--profile", name, "--output", "json", timeout=900)
    stop["v"] = True
    return rec


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "k2"
    one = name.endswith("1")
    note = f"prompt ~{TARGET} tokens, timing on" + (f", {EXTRA}" if EXTRA else "")
    res = bench(name, one, True, True, note)
    print(json.dumps(res, ensure_ascii=False))
