# Operating manual for the agents in this run directory

You are one stage of a multi-agent pipeline that builds an electromagnetic-field surrogate **from scratch** for a single scattering target. Every stage runs as an autonomous Codex agent inside **this run directory**. Read `PROBLEM.md` for the task. Your **specific role and deliverables for this invocation are given in the prompt** you were launched with — do exactly that role, nothing more.

## Execution model

- You have **full autonomy and no sandbox**. You may create/edit files, install Python packages (`pip install ...`), and run code and training jobs as needed to complete your role.
- All computation is **CPU-only**. Keep runs tractable: prefer efficient, vectorized code; size models and epoch counts sensibly for CPU.
- Be decisive and finish the job. Do not ask for confirmation — there is no human in the loop. Run everything you write and verify it actually produced the expected outputs before you stop.

## Hard rules: stay in this directory (isolation)

- **Work only inside this run directory.** Everything you need is here: `PROBLEM.md`, `dataset.json`, the raw data under `data/`, and the blackboard files written by earlier stages (`LEDGER.md`, `ledger.json`, `PLAN.md`).
- **Do NOT read, search, or copy any files outside this directory.** In particular there exists a separate, hand-written reference solution elsewhere in the wider repository — you must never look for it or use it. The whole point is that *you* derive the solution yourself from the data and the analysis. Treat anything outside this directory as off-limits.
- Do not import code from sibling run directories or parent folders. This run must be self-contained.

## The blackboard (how stages communicate)

Stages share state through files in this directory. Each stage reads what earlier stages wrote and appends its own:

| File | Written by | Purpose |
|---|---|---|
| `LEDGER.md` + `ledger.json` | Analyst | Genuine, computed understanding of the geometry and the field data, plus figures in `figures/`. |
| `PLAN.md` | Architect | The concrete, from-scratch build spec for the surrogate, justified by the ledger. |
| `workspace/` + `metrics.json` | Engineer | The model code, trained weights, evaluation numbers, and prediction plots. |
| `REPORT.md` | Reporter | Final results table + narrative. |

## Honesty rules (important)

- The understanding layer must be **real**: every number in the ledger must be **computed from the actual data** you loaded. Do **not** invent or hand-wave physics quantities. If you did not measure it, do not claim it.
- Prefer concrete, verifiable statements ("the 40 highest-|E| points all lie within radius 0.05 of the point (x,y,z) = …, which is the sharp end of the geometry") over vague ones ("the field is large near the tip").
- If something fails, say so plainly in your output and record what happened — do not paper over it.

## Logging

- Save a `log.txt` (or stage-named log) in this directory capturing what you did: key decisions, commands run, timings, iteration/loss traces during training, and final metrics.
- Print progress to stdout as you go (the orchestrator captures it).

## Plots

- When your role calls for figures, write them as `.png` files into `figures/` in this directory. Use a non-interactive Matplotlib backend (`matplotlib.use("Agg")`). Give them descriptive filenames.
