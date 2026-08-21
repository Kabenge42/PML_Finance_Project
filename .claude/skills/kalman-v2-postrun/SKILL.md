---
name: kalman-v2-postrun
description: Analyse the latest exported Kalman v2 run and update the published post-run artifact. Summarises gates and diagnostics, compares the expected-return and risk ladder (implied_upside vs expected_return_kalman vs er_mean vs risk_adj_return and the risk-normalised columns), and revises the recommendations where the numbers warrant it. Use after `python pymc_kalman_filter_pt_v2.py --write` has exported a new run, or when asked to refresh the Kalman v2 analysis artifact.
---

# Kalman v2 post-run analysis

Turn one exported run into a reviewed update of the published artifact:
**https://claude.ai/code/artifact/8f2b8e7c-f299-4c56-84ce-8569e53ad168** ("The Second Moment").

Three deliverables, in order: **(a)** gates and diagnostics, **(b)** the
return/risk ladder against prior runs, **(c)** revised recommendations.

## Before anything else — two facts about this pipeline

1. **The database holds exactly one run.** The v2 analytics tables are
   DROP-and-RECREATE. The moment a new run exports, the previous one is gone —
   and the local CSVs under `pymc_kalman_filter_pt_v2_results/` are often
   staler still. `analysis/kalman_v2_run_history.json` is the only durable
   cross-run record. Never try to query "last run" from the database.
2. **The gate report may not exist.** It is written by `export_analytics` to
   `analytics.09_gate_report_v2` and by `summarise()` to
   `<results>/09_gate_report_v2.csv`, but runs exported before 2026-08-21 have
   neither. When it is missing the script reconstructs five gates and names the
   sixteen it cannot see. **Say which you are looking at.** A reconstructed
   subset presented as a full report is how a pass-through once cleared 19 of 21.

## Step 1 — Run the extractor

```powershell
. .\set_env.ps1
python .claude/skills/kalman-v2-postrun/analyze.py --append
```

`--append` records the run in the history (idempotent on `run_id`, so a re-run
never double-records). Drop it for a dry look. `--json` gives the machine-readable
form when you need a field the report does not print.

The script owns extraction; you own judgement. **Do not recompute its statistics
by hand** — that is how two runs end up compared on different definitions. If a
number you need is missing, add it to the script so the next run has it too.

> Worked example of why: the artifact's three-run table lists "coverage
> gradient" as 1.53x / 1.56x / 1.99x and reads it as an improvement. The first
> two are the gate's own statistic (mean `er_sd` over buckets `[0,3,8,20,inf]`);
> the third was computed ad hoc as a median of `expected_upside_sd` over two
> buckets. On run `78801513e2cf` those two definitions give **1.53x and 1.98x on
> the same data**. The series compares different measurements and the
> improvement is an artefact. Fix this row when you next touch the table.

## Step 2 — Read the run before writing anything

Check `deltas` first: `[MATERIAL]` entries are the story, everything else is
housekeeping. Then sanity-check the three things that decide whether the run is
usable at all:

- **Pass-through alarm.** Spearman(`expected_return_kalman`, `implied_upside`)
  at or above **0.995** means the screen is a consensus sort and the model is not
  filtering. This is the failure of run `49e84d7e9d59` and it is the single most
  important thing on the page.
- **Convergence.** max R-hat < 1.01, min bulk ESS >= 400. Pinned parameters
  (`sd == 0`) are excluded by construction — `alpha_time[t3]` and
  `sigma_time[t3]` report R-hat NaN because they are concatenation anchors, not
  because anything is wrong.
- **Degeneracy.** `prob_pos` pinned <= 60%, and `p_upside_pos_cond` — the actual
  ranking column — comfortably interior.

## Step 3 — The three deliverables

### (a) Gates and diagnostics

Report pass/warn/fail with the gate's own value and threshold. Then, in the same
breath, **regenerate the "what this analysis can and cannot see" callout** from
`gates.unavailable`. Never carry the previous run's version forward — the whole
point is that it changes when the gate table appears.

Add convergence (worst R-hat and its parameter, min bulk and tail ESS), the
variance decomposition (`w_level` / `w_state` / `w_obs`, `rho_inf`,
`ou_length_scale_days`, `nu`), the drift column count and which betas straddle
zero, and the group-effect set.

### (b) The return / risk ladder

The comparison walks from consensus outward, and **the unit column is
load-bearing**: `implied_upside`, `expected_return_kalman`, `er_mean` and
`er_p50` are raw decimal returns; `risk_adj_return`, `expected_sharpe_ratio` and
`reward_to_cvar` are dimensionless; `p_upside_pos_cond` is a probability.
Rendering a Sharpe of 1.02 as "102%" is the exact confusion the analytics DDL's
`COMMENT ON COLUMN` convention exists to prevent.

What the reader needs from it:

- **Where the model departs from consensus.** The point estimate usually will
  not (rho ~0.99); the risk-normalised columns do (rho ~0.72-0.76). That gap is
  the model's actual contribution and should be stated as such.
- **The shrinkage triple** — OLS slope and intercept, sd ratio, median absolute
  revision in pp — which together separate calibration from disagreement.
- **The two dispersions.** `expected_upside_sd` is estimation uncertainty about
  a point; `er_sd` is outcome uncertainty. Their ratio was 40x on the
  pass-through run and ~4.7x now.
- **The book.** Effective N, sector and region concentration, weighted `cvar05`
  and `er_p05`, and how many of the 25 names carry a positive 5% tail against
  the universe share. A long book with a positive expected shortfall is not
  reporting tail risk.

Column names differ between frames — `expected_upside` / `expected_return_kalman`,
`cvar05` / `cvar_5pct_kalman`, `exp_vol` / `expected_vol_kalman`, `starr` /
`reward_to_cvar`. The script maps them; quote the analytics-table names, since
those are what a dashboard reader sees.

### (c) Recommendations

Re-rank; do not append. For each existing item ask: has this run's evidence
strengthened it, weakened it, or resolved it? Mark resolved items **shipped with
the measured effect**. Add a new item only when this run's own numbers support
it — not because it seems like a good idea.

## Non-negotiables when editing the artifact

These are what make an update trustworthy rather than merely current.

- **Verify before asserting.** Any claim about what the code does gets grepped
  first. Two claims in this artifact were wrong because they were reasoned from
  plausibility: `run_model_comparison` "already exists" (it existed in v1 only),
  and a feature substitution proposed on coverage grounds that measured
  0.008-0.054 against the 0.116-0.173 it would have replaced.
- **Correct, don't overwrite.** When new measurement contradicts a standing
  claim, mark the correction in place — what was claimed, what was measured,
  what changed. The existing "corrections the implementation forced" and
  "Correction — the source had already gone one step further" callouts are the
  pattern. Silently editing a wrong number away destroys the reader's ability to
  trust the ones that were right.
- **Run-specific sections are immutable.** A table describing run X describes
  run X. Footnote what has since changed; never restate old figures as current.
  The drift-coefficient table still says "ten features" for `37e6d8966250`
  because ten is what that run fitted.
- **Say what you cannot see**, every time, from this run's `gates.unavailable`.
- **Comparison table: anchor + last three.** Always keep `49e84d7e9d59` as the
  pass-through baseline every later figure is judged against, plus the three most
  recent runs. Dropped columns stay in the history JSON.

## Publishing

1. `WebFetch` the artifact URL to recover the current HTML (it saves a local
   copy; work from that).
2. Edit the local file. **Reuse the existing design system verbatim** — the
   teal/ochre tokens, Archivo / Source Serif 4 / JetBrains Mono, and the
   `.card` / `.tw` / `.gate` / `.rec` / `.callout` components. This is an update
   to a document the reader knows, not a redesign; only load `artifact-design` if
   the structure is genuinely being reworked.
3. Validate before publishing: tag balance, no hardcoded hex outside the token
   block, no undefined CSS vars, and every table's `<td>` count matching its
   `<th>` count.
4. Republish with `url` set to the artifact URL so it keeps the same link. Keep
   the title (**The Second Moment**) and the favicon stable.

## Reference

- `CLAUDE.md` — the Bayesian-workflow stage contract, the raw-decimal unit
  convention, and the `pymc_role` / catalogue resolution chain.
- `CHANGELOG.md` — the two `[Unreleased]` Kalman v2 entries carry the reasoning
  behind `forecast_error_multiplier`, the `shrinkage_slope` tightening and the
  variance-simplex design. Read them before questioning a default.
- `pymc_kalman_filter_pt_v2.py` — `GATE_CATALOGUE` explains what every gate is
  *for*, which is usually the sentence the artifact needs.
- Do **not** write CHANGELOG entries from this skill. A release note is a
  judgement about what shipped, not a by-product of one run's numbers.
