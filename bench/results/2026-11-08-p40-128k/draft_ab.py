import json, os, re, subprocess, sys, time, urllib.request

CTRL = r"F:\Strata\control"
PY = os.path.join(CTRL, "python", "python.exe")
BASE = json.load(open(r"F:\Strata\install\strata-iq3_s.json", encoding="utf-8"))
BENCH = r"F:\tmp-strata-bench"
PORT = 8187

ENV = dict(os.environ, STRATA_PROFILE_DIR=os.path.join(BENCH, "profiles"),
           STRATA_RUNTIME_DIR=os.path.join(BENCH, "runtime"),
           STRATA_COMPONENTS_DIR=r"F:\Strata\components")

UNIT = ("The quick brown fox jumps over the lazy dog while the engineer measures memory bandwidth, "
        "cache behaviour and the cost of every hand-off between two graphics cards. ")

ARMS = [("a8k_mtp", 8000, "mtp"), ("a8k_dflash2", 8000, "dflash2"),
        ("a192k_mtp", 192000, "mtp"), ("a192k_dflash2", 192000, "dflash2")]


def profile(name, drafting):
    cfg = json.loads(json.dumps(BASE))
    cfg["port"] = PORT
    cfg["api_key"] = "none"
    cfg["log"] = os.path.join(BENCH, name + ".log")
    try:
        os.remove(cfg["log"])          # the engine appends: a stale line would be read as this run's
    except OSError:
        pass
    # always explicit: an absent key means "auto", and auto picks DFlash2 on this model
    cfg["drafting"] = drafting or "auto"
    cfg.setdefault("env", {})
    cfg["env"].update({"STRATA_DENSE_F32": "1", "STRATA_SELECT_SIMT": "1", "STRATA_PATTN_SM60": "1"})
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


def one(name, length, drafting):
    print(f"\n=== {name} ({drafting or 'MTP'})", flush=True)
    profile(name, drafting)
    rc, out = run("runtime", "start", "--profile", name, "--wait", "900", "--output", "json")
    if rc != 0:
        print("start failed", out[-400:], flush=True)
        return {"name": name, "error": out[-300:]}
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            break
        except Exception:
            time.sleep(2)
    rec = {"name": name, "drafter": drafting or "mtp", "asked": length}
    try:
        r = ask(UNIT * max(1, length // 30), 256)
        tm = r.get("timings", {})
        rec.update({"prompt_tokens": r.get("usage", {}).get("prompt_tokens"),
                    "prompt_tps": round(tm.get("prompt_per_second") or 0, 1),
                    "decode_tps": round(tm.get("predicted_per_second") or 0, 1),
                    "generated": r.get("usage", {}).get("completion_tokens")})
    except Exception as e:
        rec["error"] = str(e)
    run("runtime", "stop", "--profile", name, "--output", "json", timeout=900)
    # the engine's own line: "drafts accepted X of Y"
    try:
        txt = open(os.path.join(BENCH, name + ".log"), encoding="utf-8", errors="replace").read()
        m = re.findall(r"drafts accepted (\d+) of (\d+)", txt)
        if m:
            a, b = m[-1]
            rec["drafts"] = f"{a}/{b}"
            rec["acceptance"] = round(100.0 * int(a) / int(b), 1) if int(b) else 0.0
        rec["drafter_log"] = [ln for ln in txt.splitlines()
                              if ("drafter on" in ln or "draft layer on" in ln or "MTP layer" in ln
                                  or "Drafting:" in ln or "no MTP" in ln)][:3]
    except Exception:
        pass
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    return rec


if __name__ == "__main__":
    want = set(sys.argv[1:])
    arms = [a for a in ARMS if not want or a[0] in want]
    res = []
    for name, length, drafting in arms:
        res.append(one(name, length, drafting))
        json.dump(res, open(os.path.join(BENCH, "draft_ab.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\n== summary")
    for r in res:
        print(json.dumps(r, ensure_ascii=False))
