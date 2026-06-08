#!/usr/bin/env python3
"""Multi-agent orchestrator for the EM-field surrogate problem.

A Python DAG that launches four role-specialized **Codex CLI agents** in sequence
for each scattering target, passing a shared blackboard (files in the run dir)
between them:

    Analyst  ->  Architect  ->  Engineer  ->  Reporter

Every stage is an autonomous `codex exec` invocation run with
`--dangerously-bypass-approvals-and-sandbox` (full autonomy, no sandbox), so the
agents can install dependencies, write code, and train models on their own. The
surrogate is built FROM SCRATCH: each run directory is staged with the raw data
only — the published reference solution is never exposed to the agents.

The pipeline is geometry-agnostic. The same four stages run for sphere, almond,
or any new target added to `datasets.py`; the only thing that differs is the
measured ledger the Analyst produces. That is the generalization story.

Usage:
    codex login                                  # one-time auth (subscription)
    python orchestrate.py --smoke --geometry sphere
    python orchestrate.py --smoke --geometry almond
    python orchestrate.py --full                 # all registered geometries, full settings
    python orchestrate.py --full --geometry sphere
    python orchestrate.py --smoke --geometry sphere --fresh   # rebuild the run dir
"""

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import datasets
import prompts

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"

# Stage order. Each entry: (key, prompt_builder, label).
STAGES = [
    ("analyst", prompts.analyst_prompt, "Analyst  (geometry + data understanding)"),
    ("architect", prompts.architect_prompt, "Architect (geometry-adaptive design)"),
    ("engineer", prompts.engineer_prompt, "Engineer  (build + train + evaluate)"),
    ("reporter", prompts.reporter_prompt, "Reporter  (results + narrative)"),
]

# Per-stage wall-clock timeouts (seconds). Engineer gets far more in --full.
TIMEOUTS = {
    "smoke": {"analyst": 1800, "architect": 1200, "engineer": 3600, "reporter": 1200},
    "full": {"analyst": 2400, "architect": 1800, "engineer": 28800, "reporter": 1800},
}

# Hint passed to the agents for the training budget (mode-dependent).
EPOCH_HINT = {"smoke": "300", "full": "30000"}

# How many frequencies to keep in a smoke run.
SMOKE_N_FREQS = 10


def check_codex():
    if shutil.which("codex") is None:
        sys.exit(
            "ERROR: `codex` CLI not found on PATH.\n"
            "Install it (`npm install -g @openai/codex`) and run `codex login`."
        )


def stage_run_dir(spec: dict, mode: str, fresh: bool) -> Path:
    """Create runs/<geometry>/ as a self-contained, data-only sandbox.

    Copies AGENTS.md + PROBLEM.md in, writes dataset.json, and copies ONLY the
    selected raw data files into data/. The published model code is never placed
    here, which is what keeps the solve honest ("from scratch").
    """
    run_dir = RUNS_DIR / spec["name"]
    if fresh and run_dir.exists():
        shutil.rmtree(run_dir)
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "workspace").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    # Operating manual + task spec, copied so the run dir is self-contained.
    shutil.copy(ROOT / "AGENTS.md", run_dir / "AGENTS.md")
    shutil.copy(ROOT / "PROBLEM.md", run_dir / "PROBLEM.md")

    # Stage the raw data (copy, so no path points back out of the run dir).
    src = Path(spec["data_dir"])
    data_dst = run_dir / "data"
    wanted = set(spec["e_files"]) | set(spec["h_files"])
    have = {p.name for p in data_dst.glob("*")}
    if wanted - have:  # (re)stage only if something is missing
        for name in wanted - have:
            shutil.copy(src / name, data_dst / name)
    # Drop any stale files from a previous, larger staging.
    for p in data_dst.glob("*"):
        if p.name not in wanted:
            p.unlink()

    # dataset.json — what the agents consume. Note data_dir is the LOCAL ./data;
    # the absolute source path is deliberately not leaked into the run dir.
    spec_local = dict(spec)
    spec_local["data_dir"] = "data"
    spec_local["mode"] = mode
    (run_dir / "dataset.json").write_text(json.dumps(spec_local, indent=2))
    return run_dir


def load_state(run_dir: Path) -> dict:
    f = run_dir / "state.json"
    if f.exists():
        return json.loads(f.read_text())
    return {"completed": []}


def save_state(run_dir: Path, state: dict):
    (run_dir / "state.json").write_text(json.dumps(state, indent=2))


def run_codex(prompt: str, run_dir: Path, log_path: Path, timeout: int):
    """Launch one autonomous Codex agent rooted at run_dir, tee output to a log."""
    inner = (
        "codex exec --dangerously-bypass-approvals-and-sandbox "
        "--skip-git-repo-check "
        f"--cd {shlex.quote(str(run_dir))} "
        f"{shlex.quote(prompt)}"
    )
    script = "set -o pipefail; { " + inner + " ; } 2>&1 | tee " + shlex.quote(str(log_path))
    subprocess.run(["bash", "-c", script], timeout=timeout, check=True)


def run_geometry(name: str, mode: str, fresh: bool):
    spec = datasets.build_spec(name)
    if mode == "smoke":
        spec = datasets.subsample_for_smoke(spec, k=SMOKE_N_FREQS)

    run_dir = stage_run_dir(spec, mode, fresh)
    state = load_state(run_dir)

    ctx = {
        "name": name,
        "mode": mode,
        "epoch_hint": EPOCH_HINT[mode],
        "n_freqs": spec["n_freqs"],
        "freq_unit": spec["freq_unit"],
    }

    print(f"\n{'=' * 70}\nGEOMETRY: {name}  |  mode: {mode}  |  "
          f"{spec['n_freqs']} frequencies\n{'=' * 70}")
    print(f"Run dir: {run_dir}")

    for key, builder, label in STAGES:
        if key in state["completed"] and not fresh:
            print(f"  [skip] {label} (already completed)")
            continue
        prompt = builder(ctx)
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = run_dir / "logs" / f"{key}_{ts}.txt"
        print(f"\n  >>> {label}\n      codex agent running... (log: logs/{log_path.name})")
        t0 = time.time()
        try:
            run_codex(prompt, run_dir, log_path, TIMEOUTS[mode][key])
        except subprocess.CalledProcessError as e:
            print(f"  [FAIL] {key} stage exited {e.returncode}. See {log_path}.")
            print("        Stopping this geometry; fix and re-run (it resumes).")
            return False
        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT] {key} stage exceeded {TIMEOUTS[mode][key]}s. See {log_path}.")
            return False
        dt = time.time() - t0
        print(f"  [done] {label}  ({dt:.0f}s)")
        state["completed"].append(key)
        save_state(run_dir, state)

    print(f"\n  Outputs for {name}:")
    for f in ["LEDGER.md", "ledger.json", "PLAN.md", "metrics.json", "REPORT.md"]:
        p = run_dir / f
        print(f"    {'OK ' if p.exists() else 'MISSING '} {f}")
    figs = list((run_dir / "figures").glob("*.png"))
    print(f"    {len(figs)} figure(s) in figures/")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_const", dest="mode", const="smoke",
                      help="fast end-to-end validation (subsampled freqs, short training)")
    mode.add_argument("--full", action="store_const", dest="mode", const="full",
                      help="full settings (all frequencies, train to convergence)")
    ap.add_argument("--geometry", default="all",
                    help="geometry name (e.g. sphere, almond) or 'all' (default)")
    ap.add_argument("--fresh", action="store_true",
                    help="rebuild the run dir and re-run all stages from scratch")
    ap.add_argument("--list", action="store_true", help="list registered geometries and exit")
    args = ap.parse_args()

    if args.list:
        for n, e in datasets.REGISTRY.items():
            print(f"  {n:10s} {e['description']}")
        return

    check_codex()
    run_mode = args.mode or "smoke"
    geometries = list(datasets.REGISTRY) if args.geometry == "all" else [args.geometry]

    ok_all = True
    for name in geometries:
        ok_all &= run_geometry(name, run_mode, args.fresh)

    print(f"\n{'=' * 70}\nPipeline finished ({run_mode}). "
          f"{'All geometries completed.' if ok_all else 'Some geometries did not complete — see logs.'}")
    print(f"Results under: {RUNS_DIR}")


if __name__ == "__main__":
    main()
