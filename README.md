# RIS-RSMA secrecy optimization: source code

This repository contains **only the Python source, tests, and instructions**.
It runs a finite-budget robust RSMA/RIS/AN design and generates eleven study
figures from freshly generated Monte Carlo trials. No trial records,
precomputed figures, manuscript text, or LaTeX source are included. All run
products go under `outputs/`, which is ignored by Git.

## Algorithm implemented

The common stream carries **public** message parts. The private streams carry
one confidential message part per user; the public rate is never counted in
confidential throughput. The model uses a conservative bound for eavesdropper
CSI uncertainty and quantized RIS phase choices.

| Code | Manuscript design step |
|---|---|
| `ris_rsma/core.py`: `channel_draw`, `aperture_subset` | Generate paired direct/cascaded channels and nested RIS aperture cases |
| `physical_rates` | Evaluate the actual conservative common, private, leakage, secrecy, power, and QoS constraints |
| `_seed_beams`, `_subproblem`, `optimize_fixed` | Feasible starts; Phase I search; MMSE-based transmit surrogate and proximal updates for a fixed RIS/AN choice |
| `screened_search` | Quantized RIS and finite AN proposals, directly checked screening, and transmit refinement of the best candidates |
| `ris_rsma/campaign.py` | Eight paired-seed experiment families and matched ablations (`no_an`, `fixed_ris`, `no_ris`) |
| `ris_rsma/plot.py` | Bootstrap aggregation, availability and paired-gain tables, and all eleven Matplotlib figures |

The code checks each retained candidate against the **direct physical-rate
constraints**, including public common-rate allocation and private-stream
confidential rates. `not_found` means the search found no verified candidate;
it is not a proof of physical infeasibility. The finite-solve implementation
does not certify global optimality or the assumptions of an exact-solve
convergence theorem.

## Install

Use Python 3.12 from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
```

On Windows activate with `.venv\Scripts\activate` and use `py -3.12` to
create the environment. `requirements.txt` pins the direct dependencies used
for the reported run: NumPy 2.3.5, CasADi 3.7.2 with IPOPT, and Matplotlib
3.10.8. The model uses one CPU process per worker; setting the two thread
variables below avoids nested BLAS parallelism.

## Reproduce the full experiment and figures

Run from the repository root. Inspect the work before launching it:

```bash
python -m ris_rsma.campaign --paper --dry-run
```

For the reference per-family draw counts, parameter settings, finite budgets,
and first seed:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m ris_rsma.campaign \
  --paper --first-seed 1000 --workers 5 --candidates 12 \
  --phase-iterations 6 --inner-iterations 4

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m ris_rsma.plot
```

The campaign runs 100 seeds for each main scenario, 50 for the power sweep,
and 40 for each of the other five sweeps: 3,080 method/condition trials in
total. This can take many CPU-hours. `outputs/raw/` receives the per-family
trial CSV and a `*_run.json` recording budgets and conditions. Interrupted
campaigns can continue with the same options plus `--resume`; incompatible
metadata is rejected. Parallel rows may arrive in a different order.

The plotting code reads the locally generated CSVs and writes eleven PDF/PNG
figures plus tab-delimited numeric curve, paired-difference, and availability
tables to `outputs/figures/`. The plotted means are paired on common channel
seeds when comparing methods. The line plots use hollow markers at simulated
operating points; bootstrap intervals are stored in the numeric curve tables
instead of appearing as shaded bands. The 95% intervals use 2,000 resamples
with seed 20261004. Most panels condition on
jointly verified draws; the confidential-goodput panel assigns zero service
to an unverified trial and reports availability separately. The convergence
panel selects the median verified channel seed and reruns a prescribed fixed
RIS/AN inner diagnostic with a 20-iteration budget.

Colors, line styles, and hollow markers are controlled centrally by
`COLORS`, `LINESTYLES`, and `MARKERS` in `ris_rsma/plot.py`.

The plotted rates should be comparable under the same code, seeds, and
dependency versions, but solver and parallel-runtime differences can change
a borderline verified candidate.
Wall-clock runtime curves are especially hardware/load dependent and cannot
be expected to match numerically across machines.

On Windows PowerShell, set the threading environment first with
`$env:OPENBLAS_NUM_THREADS="1"; $env:OMP_NUM_THREADS="1"`, then run the
`python -m ...` parts of the commands above without the Unix assignments.

## Small end-to-end check

To verify the pipeline without running thousands of trials:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m ris_rsma.campaign \
  --family blocked_power --draws 2 --workers 1 --candidates 3 \
  --phase-iterations 2 --inner-iterations 2
python -m ris_rsma.plot --figure fig01_power
```

This generates an **exploratory** power plot, not the full-campaign curve. To
make other panels, run their corresponding family (see `FAMILIES` in
`ris_rsma/campaign.py`), then select a panel using `--figure NAME`. The full
`--paper` run prepares all families for the default `--figure all` option.
Use `--out` for campaign output or `--data-dir`/`--output-dir` for plotting
to work with a separate local run. Never combine budgets or parameter
settings in one output directory.

## Model and comparison boundary

The main case uses eight BS antennas, four served users, two noncolluding
eavesdroppers, 256 passive RIS elements with 2-bit phases, a 3.5 GHz carrier,
and one 360 kHz PRB. Power is mapped uniformly from a 20 MHz carrier. The
UMi path-loss/LOS equations and free-space BS-RIS link are used with explicit
independent-fading, RIS-terminal adaptation, and conditional-corridor
assumptions; this is not a spatially consistent 3GPP channel simulator.

The comparison curves use three controls implemented in this repository:
`no_an` fixes the AN power to zero, `fixed_ris` fixes every RIS phase to zero,
and `no_ris` suppresses the cascaded channel. The controls share the channel
draws, traffic model, and finite search budgets of the joint design.

## Citation

The manuscript citation will be added once it is publicly available. Until
then, cite the repository URL together with the release tag or commit hash
used to generate your results. The code can be made public before the
manuscript is posted.

## License

No license is granted by this archive; add a license chosen by the rights
holder before making a public GitHub release.
