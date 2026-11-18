"""MTP against DFlash2 at every tier: prefill and decode in one table.

Reuses what has already been measured (sweep.json for DFlash2, draft_ab.json for the 8K/192K
pairs) and only runs the missing (drafter, tier) cells.
"""
import json, os, re, subprocess, sys, time, urllib.request

CTRL = r"F:\Strata\control"
PY = os.path.join(CTRL, "python", "python.exe")
BASE = json.load(open(r"F:\Strata\install\strata-iq3_s.json", encoding="utf-8"))
BENCH = r"F:\tmp-strata-bench"
PORT = 8188
OUT = os.path.join(BENCH, "draft_sweep.json")

ENV = dict(os.environ, STRATA_PROFILE_DIR=os.path.join(BENCH, "profiles"),
           STRATA_RUNTIME_DIR=os.path.join(BENCH, "runtime"),
           STRATA_COMPONENTS_DIR=r"F:\Strata\components")

UNIT = ("The quick brown fox jumps over the lazy dog while the engineer measures memory bandwidth, "
        "cache behaviour and the cost of every hand-off between two graphics cards. ")

TIERS = [192000, 200000, 256000, 300000, 350000, 393000, 450000, 512000]
DRAFTERS = ["mtp", "dflash2"]
TAG = {"mtp": "m", "dflash2": "d"}


def seed():
    """(drafter, asked) -> record, from the runs already done."""
    out = {}
    try:
        for r in json.load(open(os.path.join(BENCH, "sweep.json"), encoding="utf-8")):
            if r.get("prompt_tps") and not r.get("error"):
                # the sweep arms did not record acceptance: read it back from their own log
                acc = dr = None
                try:
                    txt = open(os.path.join(BENCH, "s%dk.log" % (r["asked_tokens"] // 1000)),
                               encoding="utf-8", errors="replace").read()
                    m = re.findall(r"drafts accepted (\d+) of (\d+)", txt)
                    if m:
                        a, b = m[-1]
                        dr, acc = f"{a}/{b}", round(100.0 * int(a) / int(b), 1) if int(b) else 0.0
                except Exception:
                    pass
                out[("dflash2", r["asked_tokens"])] = {
                    "drafter": "dflash2", "asked": r["asked_tokens"],
                    "prompt_tokens": r.get("prompt_tokens"), "prompt_tps": r.get("prompt_tps"),
                    "decode_tps": r.get("decode_tps"), "wall_s": r.get("wall_s"),
                    "drafts": dr, "acceptance": acc, "source": "sweep.json"}
    except Exception:
        pass
    try:
        for r in json.load(open(os.path.join(BENCH, "draft_ab.json"), encoding="utf-8")):
            # only the short-context pair: those arms ran without STRATA_PREFILL_TIMING, and a
            # prefill number must not be compared across a different measurement setting
            if r.get("asked") not in (8000,) or not r.get("prompt_tps") or r.get("error"):
                continue
                rec = {"drafter": r["drafter"], "asked": r["asked"],
                       "prompt_tokens": r.get("prompt_tokens"), "prompt_tps": r.get("prompt_tps"),
                       "decode_tps": r.get("decode_tps"), "wall_s": r.get("wall_s"),
                       "drafts": r.get("drafts"), "acceptance": r.get("acceptance"),
                       "source": "draft_ab.json"}
                old = out.get((r["drafter"], r["asked"])) or {}
                for k, v in old.items():          # keep a field the older record has and this one lacks
                    if rec.get(k) in (None, "") and v not in (None, ""):
                        rec[k] = v
                out[(r["drafter"], r["asked"])] = rec
    except Exception:
        pass
    return out


def profile(name, drafting):
    cfg = json.loads(json.dumps(BASE))
    cfg["port"] = PORT
    cfg["api_key"] = "none"
    cfg["log"] = os.path.join(BENCH, name + ".log")
    try:
        os.remove(cfg["log"])
    except OSError:
        pass
    cfg["drafting"] = drafting
    cfg.setdefault("env", {})
    cfg["env"].update({"STRATA_DENSE_F32": "1", "STRATA_SELECT_SIMT": "1", "STRATA_PATTN_SM60": "1",
                       "STRATA_PREFILL_TIMING": "1"})
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


def one(drafter, length):
    name = "%s%dk" % (TAG[drafter], length // 1000)
    print(f"\n=== {name} ({drafter})", flush=True)
    profile(name, drafter)
    rc, out = run("runtime", "start", "--profile", name, "--wait", "900", "--output", "json")
    if rc != 0:
        print("start failed", out[-400:], flush=True)
        return {"drafter": drafter, "asked": length, "error": out[-300:]}
    for _ in range(90):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            break
        except Exception:
            time.sleep(2)
    rec = {"drafter": drafter, "asked": length, "source": "new"}
    try:
        t1 = time.time()
        r = ask(UNIT * max(1, length // 30), 256)
        tm = r.get("timings", {})
        rec.update({"prompt_tokens": r.get("usage", {}).get("prompt_tokens"),
                    "prompt_tps": round(tm.get("prompt_per_second") or 0, 1),
                    "decode_tps": round(tm.get("predicted_per_second") or 0, 1),
                    "generated": r.get("usage", {}).get("completion_tokens"),
                    "wall_s": round(time.time() - t1, 1)})
    except Exception as e:
        rec["error"] = str(e)
    run("runtime", "stop", "--profile", name, "--output", "json", timeout=900)
    try:
        txt = open(os.path.join(BENCH, name + ".log"), encoding="utf-8", errors="replace").read()
        m = re.findall(r"drafts accepted (\d+) of (\d+)", txt)
        if m:
            a, b = m[-1]
            rec["drafts"], rec["acceptance"] = f"{a}/{b}", round(100.0 * int(a) / int(b), 1) if int(b) else 0.0
        rec["loaded"] = [ln.split("strata ")[-1][:90] for ln in txt.splitlines()
                         if "draft layer loaded" in ln or "DFlash2 drafter on" in ln][:1]
    except Exception:
        pass
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    return rec


if __name__ == "__main__":
    have = seed()
    want = [int(a) for a in sys.argv[1:]] or TIERS
    for L in want:
        for d in DRAFTERS:
            if (d, L) in have and have[(d, L)].get("prompt_tps"):
                print(f"reuse {d} {L}", flush=True)
                continue
            rec = one(d, L)
            if rec.get("prompt_tps"):
                have[(d, L)] = rec
            json.dump([v for k, v in sorted(have.items(), key=lambda kv: (kv[0][1], kv[0][0]))],
                      open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    rows = [v for k, v in sorted(have.items(), key=lambda kv: (kv[0][1], kv[0][0]))]
    json.dump(rows, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\n== table")
    for r in rows:
        print(f"{r['asked']:>7} {r['drafter']:<8} prefill {r.get('prompt_tps')} decode {r.get('decode_tps')} acc {r.get('acceptance')}")
