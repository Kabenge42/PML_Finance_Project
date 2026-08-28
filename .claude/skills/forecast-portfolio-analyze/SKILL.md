---
name: forecast-portfolio-analyze
description: Analyse the latest Kalman v2 forecast + decision replay as an INVESTMENT question and update the published artifact. Reads the seven replay frames (books, group posture, name actions, size-down watch, engine contrast, replay gates) and reports what the portfolio actually is — concentration, tilts, where the ranking denominator sits, how far the book departs from analyst consensus, and whether it expresses the model's own group views. Use after `python kalman_portfolio.py` has replayed a fit, or when asked to refresh "The Decision Layer" artifact.
---

# Forecast + decision replay — portfolio analysis

Turn one replay into a reviewed update of the published artifact:
**https://claude.ai/code/artifact/1dde4885-697d-4d0f-8292-ed30d76ec2a2**
("The Decision Layer").

This skill's question is **"what is this portfolio, and why does it hold what it
holds"** — not "did the fit converge". The sibling skill
[`kalman-v2-postrun`](../kalman-v2-postrun/SKILL.md) owns the fit's own gates and
diagnostics; do not duplicate them here beyond the one-line provenance strip.

## Two facts about this pipeline, before anything else

1. **The replay is not the fit.** `kalman_portfolio.py` reads a compact NetCDF
   handoff (`07_posterior/07_forecast_handoff_v2.nc`) carrying four posterior
   quantities, and stamps its *own* `run_id`. The fit's `run_id` appears only
   inside the `portfolio_handoff_provenance` gate's value string. Always name
   both — "fit X, replayed as Y" — or a later reader cannot tell which run a
   figure describes.
2. **A bare replay emits seven of nine gates.** `portfolio_book_agreement` needs
   `--rank-arms` and `portfolio_factor_sensitivity` needs `--sweep`. Those are
   precisely the two questions this layer exists to make cheap, so say when they
   were not asked rather than reporting "all gates pass".

## Step 1 — Run the extractor

```powershell
. .\set_env.ps1
python .claude/skills/forecast-portfolio-analyze/analyze_portfolio.py
python .claude/skills/forecast-portfolio-analyze/analyze_portfolio.py --json   # machine form
```

It reads CSV only — no database, no PyMC import — and honours
`KALMAN_V2_RESULTS_DIR`. **The definitions live in the script.** Do not recompute
a statistic by hand in an analysis session; if the artifact needs a number the
script does not print, add it there so the next edition uses the same definition.

## Step 2 — Read four numbers before writing a word

These decide whether the run has a story or is business as usual.

| Read | Where | What it means when it fires |
|---|---|---|
| **Effective N vs nominal** | `book.effective_n` | Cap-and-spill on a plateaued ranking fills the cap and stops. Effective N well below `n_held` means "50 names" is a label, not a portfolio. |
| **Pass-through alarm** | `consensus.spearman_model_vs_consensus` | At or above **0.995** the screen is a consensus sort. This is run `49e84d7e9d59`'s failure and it outranks everything else on the page. |
| **Denominator ratio** | `ranking.ratio_median_to_book_max` | Above ~10× means the book was selected on the *absence* of modelled risk rather than on reward per unit of it. |
| **Shared names** | `two_books.shared_names` | The fit's §10b CVaR risk book and the replay's §15e decision book are both "the book". Overlap is an empirical fact every run, and nothing declares precedence. |

## Step 3 — The five deliverables

### (a) What the portfolio actually is

Concentration first, because everything downstream is a statement about the
top few names: effective N, how many sit at the cap and what they hold between
them, the cumulative weight curve, and the smallest position. Then the tilts —
size, style, sector, country — each **against the universe share from the same
frame**, never against an external index the pipeline does not know about.

Name the top capped positions individually with sector, country, expected return,
consensus upside and analyst count. Eight names holding 80% of capital is a
readable table; "the fifty names" is not a finding.

### (b) Does the book express the model's own views?

The posture layer (§14b group signals) and the ranking layer (§15e) run off one
posterior and **never consult each other**, so their agreement is measured, not
assumed. Report `posture.sector_alignment_spearman`, then the two named cases:

- the book's largest active sector and the model's verdict on it — and if a
  verdict clears its band by a hair, say so, because a signal that would have been
  silent on a marginally different draw is concurrence, not conviction;
- the weight the book holds in sectors the model formally **underweights**.

Group verdicts use a shrunk excess, `λ = τ²/(τ²+s²)`. A large raw excess that
receives no verdict is the shrinkage working — quote the raw value, λ and the
shrunk value together or the reader cannot see it happen. Note that this is the
*inverse* of what the ranking denominator does with the same thin evidence, which
is the one pairing the figure suite was built to show.

### (c) Why these names, mechanically

Two measurements explain most books, and they compound:

- **Where the denominator sits.** `ranking.ratio_median_to_book_max` and
  `book_max_pctile`. Both shipped ratio candidates have failed exactly here.
- **The sign between the halves.** `corr_denominator_vs_expected_return`. A
  *negative* correlation means the ratio does not trade reward against risk — it
  multiplies a high numerator by a vanishing denominator, which is why the top of
  the ranking becomes a plateau and the cut lands on a tie.

Then the consensus position: universe vs book median `implied_upside`, the book's
percentile, and how many book names are in the consensus top 50. If that last
number is small, the book is *not* simply "buy the highest target" — it is the top
decile of consensus filtered to names whose modelled downside also disappeared,
which is a different and worse thing.

### (d) The forward tail

`gvar >= ges >= gtr` by construction, so the claim is about **sign**. Report the
count of book names with a positive GVaR against the universe share, and the
weighted book values. A book whose 95%-worst modelled outcome is a gain has no
left tail; pair it with `kelly_pinned_*` (a pin means `E[log(1+f·r)]` never turned
over — a statement about the simulation's left tail, never a position size) and
with the BUY share of the action list. These are **one finding, not three**.

`kelly_unbounded` is a boolean, and `kelly_max_feasible` is NULL when it is true.
That pairing exists because the `export_finite` gate refused a write over a column
at `+inf`; never reintroduce an infinity to say "no maximum".

### (e) What to do with it

Rank by how much each changes the portfolio against what it costs to find out, and
say the cost. Most are one replay flag against a fit that already exists:
`--rank-arms all`, `--sector-cap`, `--size-down-veto`, `--sweep`. The vintage
capture is the only item that can settle any of it and it is not a flag.

## Non-negotiables when editing the artifact

- **Every figure caption states what THIS run measured.** A caption asserting a
  previous run's conclusion regardless of the data is the exact failure the replay
  layer exists to prevent — `plot_denominator_sanity` carries a conditional
  headline for this reason. Copy that discipline into prose.
- **Verify before asserting.** Grep any claim about what the code does. Do not
  carry a number forward because it was true last edition; `kelly_pinned_universe`
  moved from 89.3% to 98.2% to 89.6% across three runs.
- **Correct in place, don't overwrite.** When new measurement contradicts a
  standing claim, mark what was claimed, what was measured and what changed.
- **Raw decimals everywhere; percent only at the rendering boundary.** Returns
  (`expected_return`, `er_mean`, `gvar`) are decimals; `reward_to_downside` and
  `starr` are dimensionless ratios; `p_upside_pos_cond` is a probability. Rendering
  a ratio of 2402 as a percentage is the confusion the analytics DDL's
  `COMMENT ON COLUMN` convention exists to prevent.
- **Model-implied screens, not investment advice.** Every number scores the model
  against the analyst trail it was fitted to. Keep that sentence in the footer.

## Charts

`probabilistic_ml_model/visualizations/kalman_portfolio_viz.py` is the reference for what each panel is *for* and how it
must be coloured — read its module docstring before designing a new one. The
constraints that carry over to hand-authored figures in the artifact:

- The categorical set is capped at **three** validated slots (`#56b4e9`,
  `#ffb000`, `#cc79a7`) — exactly the three ranking arms. A fourth series folds
  into "Other"; it is never a generated hue.
- **Sign is never a categorical slot.** Over/under-zero uses a diverging scale and
  a zero rule; a three-state verdict is carried by position and label, because a
  verdict painted on a signed axis encodes the same thing twice.
- Summary statistics annotated on a decimated panel are computed on the **full**
  frame, and the sampled count goes in the title.
- Densities are pre-binned, ECDFs gridded, scatters decimated. A single
  prior-predictive figure once reached 207.7 MB by ignoring this.

The artifact's own charts are static inline SVG reductions of that suite — theme
them through the page's CSS tokens (`var(--accent)`, `var(--amber)`,
`var(--rose)`, `var(--ink-2)`), never with literal hex, or one theme renders the
other theme's ink.

## Publishing

1. `Artifact` with `action: "read"` and the URL to recover the current HTML, then
   read the saved local copy in full before editing.
2. **Reuse the existing design system verbatim** — the teal/slate tokens, IBM Plex
   Sans Condensed / Serif / Mono, and the `.card` / `.tw` / `.pill` / `.finding` /
   `.chart` / `.delta` components. This is an update to a document the reader
   knows. Only load `artifact-design` if the structure is genuinely being reworked.
3. Validate before publishing: tag balance, no hardcoded hex outside the token
   block, no colour defined only inside a `@media` or `[data-theme]` block, and
   every table's `<td>` count matching its `<th>` count.
4. Republish with `url` set to the artifact URL so it keeps the same link. Keep the
   title (**The Decision Layer**) and the favicon stable.

## Reference

- `kalman_portfolio.py` — the replay itself; `RANKING_RULES` and the gate
  catalogue explain what each arm and gate is *for*.
- `probabilistic_ml_model/visualizations/kalman_portfolio_viz.py` — panel intents, the colour argument, the payload
  budget.
- `probabilistic_ml_model/pymc_models/RiskBookModel.py` — `MIN_RATIO_DENOMINATOR`,
  `MIN_TAIL_RISK`, and why `tail_risk` lost its `expected_upside` leg on the
  return-draw path.
- `CLAUDE.md` — the raw-decimal unit convention, the export-gate rules, and
  `export_layout` (never build a result path by hand).
- Do **not** write CHANGELOG entries from this skill. A release note is a
  judgement about what shipped, not a by-product of one replay's numbers.
