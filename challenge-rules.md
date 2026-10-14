## Challenge rules

Compiled from the Codabench forum snapshot of 2026-10-07 ([forum/index.md](forum/index.md)). Organizer posts are pinned; participant operational threads are omitted. Where an organizer later restated a rule, the later text is the one below.

### Submission formats and award rules

**Predictions.** One Codabench upload carries both machines. In Development the files are `diii_d_public_test.npz` and `mast_public_test.npz`. In the Final phase they are `diii_d_private_test.npz` and `mast_private_test.npz`. Each sits at the root of a Hugging Face **dataset** repo (pointer route) or at the root of a zip (direct upload).

The organizers ask teams to use the **Hugging Face pointer route**: a few-hundred-byte `submission_pointer.zip` whose only file is `manifest.json`

```json
{"repo_id": "your-username/fusion-eq-predictions",
 "revision": "0123456789abcdef0123456789abcdef01234567",
 "token": "hf_..."}
```

`revision` must be the full 40-character commit SHA — not `main`, not a tag. Direct `.npz` upload still scores identically; it is slower (~1 hour transfer on Codabench storage) and counts against the 15 GB Codabench quota. Pointer is a request, not a rule change ([thread 2659](https://www.codabench.org/forums/17429/2659/)).

**Reproduction package (Final, required).** Every Final-phase submission must give, at submission time, a reproduction-package reference in the fact-sheet **Code URL** field ([thread 2807](https://www.codabench.org/forums/17429/2807/)):

- a Git repository at a **pinned commit SHA** on any public forge, or
- a versioned archive on any public host, identified by **SHA256** (`url` then `sha256:<hash>`).

The package must contain: model source; a pinned environment; the trained weights used for that submission; a **training entrypoint that rebuilds from scratch**; an inference entrypoint that regenerates the submitted predictions; a short **data-provenance statement** listing every external dataset and pre-trained model, per machine; and a **training compute budget** (hardware and wall-clock). A container image is welcome, not required. You may ship generated synthetic data with SHA256 checksums if the generator, config and seeds are also in the package ([thread 2827](https://www.codabench.org/forums/17429/2827/)).

**Who they rebuild.** Top five entries per challenge: retrain from the training entrypoint, re-score the rebuilt model, and check that the shipped weights reproduce the submitted predictions. An entry that cannot be rebuilt stays on the leaderboard but is **not award-eligible**. An inaccurate provenance statement disqualifies the entry on its own. Rebuild tolerance (how close the rebuilt score must be) was promised before the Final opens on 19 October and is not in this snapshot.

**Prize paperwork.** The binding prize condition: the **top three** teams per award category must release source under an OSI-approved licence and file a 1–2 page methods report before prizes are paid.

**What you must not submit.** Predictions you did not produce. AI coding assistants, including agentic ones, may be used for any part of an entry if the workflow is disclosed in the methods report and the entrant takes responsibility ([thread 2807](https://www.codabench.org/forums/17429/2807/)).

**Awards (what the files are scored for).**

| Award | Judged on | Score |
|---|---|---|
| Award #1 | DIII-D blind private fold | composite $S$ |
| Award #2 | MAST / DIII-D transfer | $G_{\mathrm{ratio}}=S_{\mathrm{MAST}}/S_{\mathrm{DIII\text{-}D}}$, gated |
| Honorable mentions (four) | MAST blind private fold | best $R^2\psi$; best $R^2\{q_{95},\beta_N\}$; best Consistency; lowest $D_{\mathrm{LCFS}}$ |

A MAST rule-3 breach costs Award #2 and the honorable mentions, not Award #1. A provenance or methods report that **misstated** what was used disqualifies the team from every award ([thread 2827](https://www.codabench.org/forums/17429/2827/)).

**Challenge 2 gate.** Only entries with DIII-D composite $S\ge 0.85$ are ranked on Challenge 2. Below the gate the scorer (v3.2.0) reports $G_{\mathrm{ratio}}=0$. The retired gate was $R^2\psi>0.6$ on DIII-D. $S\ge 0.85$ implies $R^2\psi\ge 0.727$. Challenge 1 is unaffected ([thread 2711](https://www.codabench.org/forums/17429/2711/)).

**How Award #2 is computed when you have several Final files.** $S_{\mathrm{MAST}}$ comes from the submission that counts for Award #2. The denominator $S_{\mathrm{DIII\text{-}D}}$ is the team’s **best** DIII-D composite across all successfully scored Final submissions, and the $S\ge 0.85$ gate applies to that best score. A weaker DIII-D model cannot raise Award #2 ([thread 2827](https://www.codabench.org/forums/17429/2827/)).

**Tie-breaks** on composite $S_{\mathrm{model}}$: first $R^2\psi$, then $D_{\mathrm{LCFS}}$ (lower is better). Honorable mentions are compared at full precision; exact ties are shared.

**What is not scored.** `dsep` is context only — not a scored term. Consistency covers **seven** $\psi$-derived scalars: $R_{\mathrm{axis}}$, $Z_{\mathrm{axis}}$, $\kappa$, $\mathrm{tri_{top}}$, $\mathrm{tri_{bot}}$, volume, $\ell_i$ — not eight ([thread 2807](https://www.codabench.org/forums/17429/2807/)).

**Per-shot Detailed Results.** Rows are keyed by submission key (`shot_0042`), not machine shot number (scorer v3.3.0). In the Final (blind) phase the per-shot table is withheld; aggregates, histograms and the scalar breakdown remain.

**Winners.** The **Final phase, scored on the blind private fold**, decides the winners. The Development leaderboard is a practice board and is not re-scored. A sentence that the best Development entry formed the ranking basis was an error and is being corrected ([thread 2807](https://www.codabench.org/forums/17429/2807/)).

## Training-specific restrictions

Machine-specific pipelines are allowed: the DIII-D file and the MAST file may come from different models, disclosed in the methods report. The constraint is what the **MAST path** may have seen. “MAST path” means any component whose output influences MAST predictions — feature extractors, bases, priors, normalizers, and the choice of hyperparameters or checkpoints ([thread 2807](https://www.codabench.org/forums/17429/2807/)).

**Rule (23 September 2026, effective for Final and all award decisions).** Equilibrium reconstructions of MAST or MAST-U from any source — the FAIR-MAST archive, mirrors, or any derivative, including model weights trained on them — may not be used to train, fine-tune, calibrate, validate, or select any model or component that produces MAST predictions. That covers hyperparameter search and checkpoint selection, and it covers flux maps, LCFS contours and EFIT-derived scalars alike.

| Data source | In the MAST path | In the DIII-D path |
|---|---|---|
| MAST/MAST-U **equilibrium reconstructions** (FAIR-MAST EFIT++ $\psi$, LCFS, EFIT-derived scalars), mirrors, or datasets derived from them | **No** | Yes, disclosed — not if the same component also produces MAST output |
| **Aggregates** of MAST equilibria (PCA bases, population statistics, prior corridors) | **No** | Yes, disclosed, same caveat |
| Pre-trained checkpoints whose training data **includes MAST equilibria** | **No** | Yes, disclosed, same caveat |
| The three labelled **MAST demo shots** in the starter kit | Visualization and format inspection only | n/a |
| **MAST inputs** shipped with the challenge (coils, $I_p$, Thomson) | Yes — inference, self-supervised pretraining, normalization, alignment, label-free test-time adaptation | Yes |
| **Public MAST inputs** not in the release (e.g. FAIR-MAST level-1 signals such as per-winding P2 currents) | Yes, disclosed | Yes, disclosed |
| MAST **machine description**: geometry, grids, wall, coil rectangles, unit conventions | Yes | Yes |
| **Other machines’** equilibria (DIII-D train split, NSTX-U, JET), OMFIT tables | Yes, disclosed | Yes, disclosed |
| **Synthetic** equilibria from your own free-boundary solver, including MAST geometry | Yes — may be calibrated against MAST *inputs*, never against MAST EFIT reconstructions | Yes |
| Your own **previous submissions** and their public-fold scores | Yes — unless the distilled source used barred data | Yes |

**Development leaderboard feedback on MAST is permitted, with disclosure.** The scores and Detailed Results the platform returns are competition feedback. Every team gets the same feedback at the same rate. You may use it to compare, tune and select MAST pipelines; say so in the methods report. What remains barred is MAST equilibrium data you can score against **yourself, offline**: archive reconstructions, the three labelled demo shots, or anything derived from them ([thread 2827](https://www.codabench.org/forums/17429/2827/)).

**Probe submissions.** You may use the score each submitted candidate receives to choose among candidates you submitted. You may not use submissions as measurements of the labels (constant, shifted or rescaled probes to read the labels). Label statistics obtained that way, and anything fitted to them (bias, scale, ensemble weights, post-processing constants found by computing scores offline), count as MAST equilibrium data. None of it may be part of a Final MAST pipeline. Not retroactive; Development entries are not judged. Leave probe-derived numbers out of the Final pipeline and describe what you did in the methods report ([thread 2827](https://www.codabench.org/forums/17429/2827/), follow-up 5 October).

**Envelope mask.** `fusion_scoring/masks/mast_envelope.npz` is the union of every ground-truth LCFS in the MAST corpus, dilated by two cells — an aggregate of MAST labels. Running the official derivation (the scorer’s LCFS search) on your own predictions is fine. Using the mask as a model input, feature, loss mask or prior for MAST predictions is not. MAST wall and limiter geometry is permitted and serves the same purpose.

**Compute ceiling (rebuild).** Training must be reproducible within **24 GPU-hours on a single 80 GB A100-class GPU**, or equivalent CPU time. Only **GPU** work counts toward that 24 hours. CPU-only data generation (for example a free-boundary solver) must be declared with hardware and wall-clock but is not capped; GPU time spent generating data does count. If the pipeline cannot fit the ceiling, say so in the package for a case-by-case check rather than automatic disqualification ([threads 2807](https://www.codabench.org/forums/17429/2807/), [2827](https://www.codabench.org/forums/17429/2827/)).

**Still permitted on the MAST path, and often needed:** the shipped MAST inputs used without targets; per-machine normalization from inputs; MAST geometry, grids, wall, coils, unit conventions; other machines’ equilibria (disclosed); your own synthetic equilibria calibrated to MAST inputs only.

**Coil-current units (authoritative statement, Data tab / dataset card).** Shaping-coil currents are **kA·turn** (DIII-D `F1A`–`F9B`, MAST `p{2-6}{l,u}`). Plasma current and DIII-D `ECOILA` / `bcoil`, MAST `sol` / `tf` / `efps` are **kA**. There is no `coil_turns` column. `sol_current` reads exactly **2×** the FAIR-MAST level-2 `pf_active` solenoid current; the organizers have not resolved which convention is correct — treat the factor as an open question if the solenoid is used quantitatively. `tf_current` and DIII-D `bcoil` have no poloidal rectangle. `efps_current` is a feed current at the P2 link board, not a coil ([thread 2807](https://www.codabench.org/forums/17429/2807/)).

**DIII-D $I_p$ timestamps (not a training ban).** On about 69% of DIII-D shots `magnetics_plasma_current_times` carries the wrong time axis. The $I_p$ **values**, all targets, `efit_times`, MAST, and scoring are unaffected. Use the starter helper `fix_d3d_ip_times(row)` per shot — do not blanket-shift. A MAST Thomson channel-alignment helper (`align_mast_thomson_core`) ships with it ([thread 2709](https://www.codabench.org/forums/17429/2709/)).

## Timeline

Dates below are those the organizer posts state. The forum snapshot is 2026-10-07.

| When | What |
|---|---|
| **October 17, 2026 at 8:00 PM EDT** | Development (phase 1) **closes**. Last submit. Codabench results tab: “Current Phase Ends” ([#17739](https://www.codabench.org/competitions/17739/#/results-tab)). Practice board. **5 submissions a day, 100 in total.** Code URL optional. MAST per-shot table uses submission keys. |
| **19 October 2026** | Final phase **opens**. Rebuild-tolerance note promised before this date. |
| Final phase | Blind private fold. **Up to three** submissions. Code URL / reproduction package **required**. Per-shot table withheld. |
| **2026-10-26 00:00 UTC** | Final phase **closes**. Award-nomination email must arrive before this instant. |

**Which Final submission counts.**

- **Default:** the most recent Final submission that **scored successfully** counts for both awards and the honorable mentions. A submission that fails to score never displaces an earlier scored one.
- **Nomination (optional):** one scored Final submission for Award #1 and a different one for Award #2. Email `fusionec26@sophelio.io` with team name and the two submission IDs before the close. Each nominated submission needs its own reproduction-package reference.
- **Platform faults.** If a Final submission sticks in the queue or fails on the organizers’ side, email them; they will remove it so the slot comes back.

Development entries are not judged retroactively under the 23 September MAST rule. Awards are decided only on the Final.

## Websites

| What | URL |
|---|---|
| Live Codabench competition | https://www.codabench.org/competitions/17739/ — **#17739**. Older cards that still say **#17456** are stale ([thread 2807](https://www.codabench.org/forums/17429/2807/)). |
| Codabench forum (this compilation’s source) | https://www.codabench.org/forums/17429/ |
| Challenge website / announcements | https://fusion-equilibrium-challenge.sophelio.io/announcements/ |
| Starter kit | https://github.com/Sophelio/fusion-equilibrium-challenge-starter |
| Starter issue that set the $S\ge 0.85$ gate | https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/7 |
| $I_p$-timestamp erratum | https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/6 ; fix notes in the starter README “Data errata v1.1.0” |
| Hugging Face token settings (pointer read token) | https://huggingface.co/settings/tokens |
| Organizer email | fusionec26@sophelio.io |

Local copies of the pinned organizer threads: [2659 pointer route](forum/threads/2659-please-submit-via-the-hugging-face-pointer-route-faster-scoring-for-everyone.md), [2709 $I_p$ times](forum/threads/2709-data-erratum-fix-diii-d-plasma-current-timestamps-scoring-unaffected.md), [2711 Ch2 gate](forum/threads/2711-rules-update-challenge-2-eligibility-gate-is-now-diii-d-composite-s-0-85-scorer-.md), [2807 MAST rule / repro / docs](forum/threads/2807-rules-update-mast-equilibrium-data-per-shot-reporting-and-reproduction-packages-.md), [2827 Final counting / probes / mentions](forum/threads/2827-clarifications-development-feedback-on-mast-which-final-submission-counts-honora.md).
