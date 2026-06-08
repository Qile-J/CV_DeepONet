# Problem: learn the incident → total electromagnetic field operator, generalizing to unseen frequencies

You are given electromagnetic scattering data for **one metallic (perfect electric conductor) target**, computed by a frequency-domain solver on a **fixed mesh** at many discrete frequencies. The fields are time-harmonic, so every field value is a **complex number** (a phasor with real and imaginary parts).

Your job is to build a **surrogate model** that, given the **incident** field as input, predicts the **total** (scattered + incident) field over the whole mesh — and crucially, that does so **at frequencies it never saw during training**. The mesh (the set of spatial points) is identical for every frequency; what changes from sample to sample is the frequency and therefore the field.

This is an **operator-learning** problem: you are learning a mapping from an input *function* (the incident field over space) to an output *function* (the total field over space), parameterized by frequency. How you represent and learn that operator is **entirely your decision** — there is no prescribed architecture. Design whatever works best for the geometry and data you are given.

## What you are NOT told (and must discover)

Nothing about the architecture, the network type, the framework, or the hyperparameters is fixed. You decide all of it, informed by the geometry/data analysis handed to you in `LEDGER.md` / `ledger.json`. Different geometries may warrant different choices — e.g. a target with a sharp tip concentrates the field there and is harder to fit near that singularity than a smooth target.

## The defining constraint: the split is over FREQUENCIES, not over space

- The mesh is **fixed**. Every frequency sample sees the **same** spatial points.
- You **train on a subset of the available frequencies** and you are evaluated on the **held-out frequencies**.
- The whole point of the surrogate is **frequency interpolation/extrapolation**: predict the field at frequencies the model never trained on.
- A standard, robust way to pick the training frequencies: take `np.linspace` across the available frequency indices for the training count, **force the two endpoints into the training set** (so you never extrapolate past the data range), then treat the remaining frequencies as the held-out test set. You may improve on this, but never let test frequencies leak into training.

## Data format (geometry-agnostic — the same layout for every target)

The raw data lives in your run directory under `data/`. There are two families of whitespace-delimited text files, one row per mesh cell:

- `eall_pec.*` (or `eall_pec_f_*.dat`) — the **E** (electric) field files, one file per frequency.
- `hall_pec.*` (or `hall_pec_f_*.dat`) — the **H** (magnetic) field files, one file per frequency.

Sort the files with a **natural-alphanumeric** sort so frequency 2 precedes frequency 10 (and `.001` precedes `.010`). The i-th E file and the i-th H file correspond to the same frequency.

Each row has **16 columns**:

| Columns (0-indexed) | Meaning |
|---|---|
| 0 | cell index |
| 1, 2, 3 | `x, y, z` coordinates of the mesh point (the **geometry**) |
| 4, 5 | Re, Im of the **total** field, component x |
| 6, 7 | Re, Im of the **total** field, component y |
| 8, 9 | Re, Im of the **total** field, component z |
| 10, 11 | Re, Im of the **incident** field, component x |
| 12, 13 | Re, Im of the **incident** field, component y |
| 14, 15 | Re, Im of the **incident** field, component z |

So per file you get, for every mesh point: the 3 complex **total**-field components (cols 4–9, your prediction **target**) and the 3 complex **incident**-field components (cols 10–15, your **input**). E files give the E components, H files give the H components — six complex field components in total: `Ex, Ey, Ez, Hx, Hy, Hz`.

The frequency associated with each file index is provided in `dataset.json` in your run directory (either explicit GHz values, or — if the original problem only used indices — the integer frequency index). Use whatever `dataset.json` gives you as the frequency coordinate.

## Metric (report this exactly)

For each of the six components, on the **held-out (test) frequencies**, report the **relative L2 error in percent**, computed per frequency and averaged over the test frequencies:

```
relL2(%) = mean_over_test_freqs(  || pred - true ||_2  /  || true ||_2  ) * 100
```

where the norm is over all mesh points (treating the complex field as a complex vector). Report it per component (`Ex … Hz`) for both the training and the held-out test frequencies, plus the mean test relL2 across the six components. Lower is better.

## Success criteria

- A trained surrogate whose **held-out-frequency** relative-L2 is low across the six components.
- Predictions that visibly track the ground truth (you will produce prediction-vs-truth plots).
- Code that is **geometry-agnostic**: it reads whatever target's data is in `data/` and adapts using the ledger — it must not hardcode anything specific to a particular target.
