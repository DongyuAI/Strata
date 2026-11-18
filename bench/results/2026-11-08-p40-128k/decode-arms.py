import json, os, subprocess, sys, time, urllib.request, urllib.error

CTRL = r"F:\Strata\control"
PY = os.path.join(CTRL, "python", "python.exe")
BASE = json.load(open(r"F:\Strata\install\strata-iq3_s.json", encoding="utf-8"))
BENCH = r"F:\tmp-strata-bench"
PORT = 8181

PROMPT = (
    "You are explaining computer hardware to a beginner. Write a careful, factual explanation of how a "
    "GPU executes a matrix multiplication, why memory bandwidth matters more than compute for token "
    "generation, and what changes when two GPUs split one model between them. Use plain words, short "
    "sentences, and one concrete example with numbers. Do not use lists longer than three items. "
) * 6


def profile(name, extra=(), replace=None, roles=None, cfg_extra=None):
    cfg = json.loads(json.dumps(BASE))
    args = list(cfg.get("args", []))
    if replace:
        out, i = [], 0
        while i < len(args):
            if args[i] in replace:
                i += 1 + len(replace[args[i]])
                continue
            out.append(args[i])
            i += 1
        args = out
    cfg["args"] = args + list(extra)
    cfg["port"] = PORT
    cfg["api_key"] = "none"
    cfg["log"] = os.path.join(BENCH, name + ".log")
    if roles:
        cfg["gpu_roles"] = roles
    if cfg_extra:
        cfg.update(cfg_extra)
    path = os.path.join(BENCH, name + ".json")
    json.dump(cfg, open(path, "w", encoding="utf-8"), indent=2)
    return path


ENV = dict(os.environ, STRATA_PROFILE_DIR=os.path.join(BENCH, "profiles"),
           STRATA_RUNTIME_DIR=os.path.join(BENCH, "runtime"),
           STRATA_COMPONENTS_DIR=r"F:\Strata\components")


def run(*a, timeout=3600):
    p = subprocess.run([PY, "strata.py", *a], cwd=CTRL, capture_output=True, text=True, timeout=timeout, env=ENV)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def save_profile(name, source):
    d = os.path.join(BENCH, "profiles")
    os.makedirs(d, exist_ok=True)
    rec = {"id": name, "overrides": {"port": PORT}, "schema_version": 1, "source_config": source}
    json.dump(rec, open(os.path.join(d, name + ".json"), "w", encoding="utf-8"), indent=2)


def start(path, name):
    save_profile(name, path)
    t0 = time.time()
    rc, out = run("runtime", "start", "--profile", name, "--wait", "900", "--output", "json")
    return rc, out, time.time() - t0


def wait_ready(limit=60):
    for _ in range(limit):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def ask(prompt, max_tokens=128):
    body = json.dumps({"model": "bench", "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": max_tokens, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
                                 headers={"content-type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read().decode())
    d["wall_s"] = round(time.time() - t0, 2)
    return d


def bench(name, path, note):
    print(f"\n=== {name}: {note}", flush=True)
    rc, out, secs = start(path, name)
    if rc != 0:
        print(f"start failed rc={rc} in {secs:.0f}s\n{out[-1500:]}", flush=True)
        run("runtime", "stop", "--profile", name, "--output", "json")
        return {"name": name, "note": note, "error": out[-600:], "start_s": round(secs)}
    print(f"started in {secs:.0f}s", flush=True)
    if not wait_ready():
        print("not ready", flush=True)
        run("runtime", "stop", "--profile", name, "--output", "json")
        return {"name": name, "note": note, "error": "not ready", "start_s": round(secs)}
    try:
        ask(PROMPT, 64)                      # warm the expert cache
        runs = [ask(PROMPT, 256) for _ in range(3)]
    except Exception as e:
        print("request failed:", e, flush=True)
        run("runtime", "stop", "--profile", name, "--output", "json")
        return {"name": name, "note": note, "error": str(e), "start_s": round(secs)}
    dec = sorted(r.get("timings", {}).get("predicted_per_second") or 0 for r in runs)
    pre = sorted(r.get("timings", {}).get("prompt_per_second") or 0 for r in runs)
    a = runs[-1]
    tm = a.get("timings", {})
    rec = {"name": name, "note": note, "start_s": round(secs),
           "prompt_tokens": a.get("usage", {}).get("prompt_tokens"),
           "out_tokens": a.get("usage", {}).get("completion_tokens"),
           "prefill_tps": round(pre[1], 1),
           "decode_tps": round(dec[1], 1),
           "decode_all": [round(x, 1) for x in dec]}
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    rc, out = run("runtime", "stop", "--profile", name, "--output", "json", timeout=600)
    print("stopped rc", rc, flush=True)
    return rec


if __name__ == "__main__":
    which = sys.argv[1:] or ["v0"]
    results = []
    for name in which:
        note = ""
        if name == "v0":
            note = "baseline: current app settings"
            p = profile("v0")
        elif name == "v1":
            note = "reserve 1152 MiB (stop the paging warning)"
            p = profile("v1", extra=["--vram-reserve-mib", "1152"], replace={"--vram-reserve-mib": ["512"]})
        elif name == "v2":
            note = "pipeline windows 2"
            p = profile("v2", extra=["--pipeline-windows", "2"])
        elif name == "v3":
            note = "main on GPU0 (x16 slot)"
            p = profile("v3", roles={"0": "main", "1": "helper"})
        elif name == "v5":
            note = "DFlash2 drafter on the split, reserve 1152"
            p = profile("v5", extra=["--vram-reserve-mib", "1152"],
                        replace={"--vram-reserve-mib": ["512"]}, cfg_extra={"drafting": "dflash2"})
        elif name == "v8":
            note = "ONE card, no DENSE_F32, DFlash2"
            p = profile("v8", extra=["--gpu", "0"],
                        cfg_extra={"gpu": 0, "gpu_roles": {"0": "main", "1": "off"}, "drafting": "dflash2"})
        elif name == "v7":
            note = "ONE card + DENSE_F32 + DFlash2 (decode)"
            p = profile("v7", extra=["--gpu", "0"],
                        cfg_extra={"gpu": 0, "gpu_roles": {"0": "main", "1": "off"}, "drafting": "dflash2",
                                   "env": {"STRATA_RAM_KEEP_GIB": "8", "STRATA_DENSE_F32": "1"}})
        elif name == "v6":
            note = "DFlash2 drafter, reserve left at 512 (the app's current setting)"
            p = profile("v6", cfg_extra={"drafting": "dflash2"})
        elif name == "v4":
            note = "context 32k, VRAM to the expert cache"
            p = profile("v4", extra=["--max-context", "32768"], replace={"--max-context": ["131072"]})
        else:
            raise SystemExit("unknown " + name)
        results.append(bench(name, p, note))
        json.dump(results, open(os.path.join(BENCH, "results.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("== summary")
    for r in results:
        print(json.dumps(r, ensure_ascii=False))
