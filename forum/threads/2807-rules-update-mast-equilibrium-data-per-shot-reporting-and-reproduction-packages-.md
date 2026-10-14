# Rules update: MAST equilibrium data, per-shot reporting, and reproduction packages for the Final phase

- Thread: https://www.codabench.org/forums/17429/2807/
- Started by: sophelio
- Date created: Sep 23, 2026
- Pinned: yes
- Posts: 1

---

## Post 1 — sophelio

*Sept. 23, 2026, 7:59 p.m.*

Dear participants,

Three updates, all effective today. Two of them change what you are allowed to do, so please read
before your next submission. Several of you raised exactly these points by email and on the starter
repo — thank you; this post is the public answer, and we are replying individually as well.

## 1. MAST equilibrium reconstructions may not be used for MAST predictions

Rule 2 permits external public tokamak archives with disclosure. That wording was written with
*other* machines in mind, and it collides with the premise of Challenge 2: MAST's withheld targets
come from a public archive, so "an external archive" and "the answer key" are the same object. We
are closing that, and we are stating plainly why.

**New rule.** Equilibrium reconstructions of MAST or MAST-U from any source — the FAIR-MAST
archive, mirrors of it, or any derivative, including model weights trained on them — may not be used
to train, fine-tune, calibrate, validate, or select any model or component that produces MAST
predictions. This covers hyperparameter search and checkpoint selection, and it covers flux maps,
LCFS contours and EFIT-derived scalars alike.

**Explicitly still permitted, and unaffected:**

- The MAST inputs shipped with the challenge, used **without targets** — including self-supervised
  or unsupervised pretraining, target-domain normalization statistics, distribution alignment, and
  label-free test-time adaptation (for example physics-residual optimization).
- Per-machine normalization computed from inputs. This remains recommended.
- MAST machine geometry, grids, wall and coil descriptions, and unit conventions.
- Equilibria from **other** machines (DIII-D, NSTX-U, JET, …) and OMFIT-produced tables.
- Synthetic equilibria from your own free-boundary solver, including a solver set up for MAST
  geometry. You may calibrate such a generator against MAST **inputs**; you may not calibrate it
  against MAST EFIT reconstructions.
- The three labelled MAST demo shots in the starter kit: format inspection and visualization only,
  as documented. Not validation, model selection, calibration or training.

**This removes a question several of you asked:** because MAST labels are out of scope entirely, you
no longer need a way to certify that an external MAST training population is disjoint from our test
folds. There was no such mechanism — test-shot identities are withheld by design — and now none is
needed.

**Effective now, for the Final phase and for all award decisions.** Development-phase entries are
not judged retroactively; the Development leaderboard stands as a practice board. What is scored for
awards is the Final phase, and award candidates must show compliance in their reproduction package
(section 3).

Machine-specific pipelines remain allowed: your DIII-D file and your MAST file may come from
different models, disclosed in the methods report. The constraint is on what the MAST path is
allowed to have seen, not on sharing one set of weights.

### Data source × permitted use (Challenge 2)

"MAST path" means any component whose output influences your MAST predictions — including feature
extractors, bases, priors, normalizers, and the choice of hyperparameters or checkpoints.

| Data source | In the MAST path | In the DIII-D path |
|---|---|---|
| MAST/MAST-U **equilibrium reconstructions** (FAIR-MAST EFIT++ ψ, LCFS, EFIT-derived scalars), mirrors, or datasets derived from them | **No** | Yes, disclosed — but not if the same component also produces MAST output |
| **Aggregates** of MAST equilibria (PCA bases, population statistics, prior corridors) | **No** — a compressed form of the withheld target | Yes, disclosed, same caveat |
| Pre-trained checkpoints whose training data **includes MAST equilibria** | **No** | Yes, disclosed, same caveat |
| The three labelled **MAST demo shots** in the starter kit | Visualization and format inspection only | n/a |
| **MAST inputs** shipped with the challenge (coils, Ip, Thomson) | Yes — inference, self-supervised pretraining, normalization, alignment, label-free test-time adaptation | Yes |
| **Public MAST inputs** not in the release (e.g. FAIR-MAST level-1 diagnostic signals such as per-winding P2 currents) | Yes, disclosed | Yes, disclosed |
| MAST **machine description**: geometry, grids, wall, coil rectangles, unit conventions | Yes | Yes |
| **Other machines'** equilibria (DIII-D train split, NSTX-U, JET), OMFIT tables | Yes, disclosed | Yes, disclosed |
| **Synthetic** equilibria from your own free-boundary solver, incl. MAST geometry | Yes — may be calibrated against MAST *inputs*, never against MAST EFIT reconstructions | Yes |
| Your own **previous submissions** and their public-fold scores (self-distillation) | Yes — ordinary own work, unless the distilled source used barred data | Yes |

## 2. Per-shot reporting in Detailed Results

A participant pointed out that the per-submission Detailed Results page listed the twenty worst
shots by their real machine shot numbers. On MAST those numbers index a public archive, so the table
was a slow leak of the fold's identity. That was our bug and we are grateful it was reported rather
than used.

As of scoring version 3.3.0:

- Per-shot rows are keyed by the **submission key** you already use (`shot_0042`), not by machine
  shot number.
- In the **Final (blind) phase the per-shot table is withheld entirely.** Aggregate scores,
  histograms and the scalar breakdown remain.

Existing submissions are not re-scored; the change applies from the next submission onward.

## 3. Reproduction packages — new requirement for the Final phase

We will validate the leading entries by rebuilding them, not merely by re-running their predictions.
To make that possible:

**Every Final-phase submission must be accompanied, at submission time, by a reproduction package
reference.** Give us either:

- a Git repository at a **pinned commit SHA**, on any public forge (GitHub, GitLab, Codeberg, an
  institutional host — no preference, and GitHub is not required); or
- a versioned archive on any public host (Zenodo, Hugging Face, institutional storage), identified
  by its **SHA256**.

Record it in the **Code URL** field of the submission fact sheet — repo URL with the pinned SHA
appended, or archive URL followed by `sha256:<hash>`. That field already exists and stays optional
during Development; it is required from the Final phase on. The Submission format tab has worked
examples of both forms.

The package must contain: model source; a pinned environment specification; the trained weights used
for the submission; a **training entrypoint that rebuilds the model from scratch**; an inference
entrypoint that regenerates the submitted predictions; and a short data-provenance statement listing
every external dataset and pre-trained model used, per machine. A container image is welcome but not
required.

**Declare your training compute budget** in the package (hardware and wall-clock). Training must be
reproducible within **24 GPU-hours on a single 80 GB A100-class GPU**, or equivalent CPU time. If your pipeline genuinely cannot fit that ceiling, say so in the package and
we will arrange a case-by-case check rather than disqualify you.

**Who we will run.** We will ask the **top five entries per challenge** for a full rebuild — retrain
from the training entrypoint, then re-score from the resulting model, and separately verify that the
shipped weights reproduce the submitted predictions. The binding prize condition remains as
published: the **top three** teams per award category must release source under an OSI-approved
licence and file a 1–2 page methods report before prizes are paid. Asking five gives us room when an
entry above cannot be reproduced.

An entry that cannot be rebuilt stays on the leaderboard but is not eligible for an award. This is
the existing reproducibility rule, made concrete.

**Tie-breaks.** Ties on the composite `S_model` are broken first by `R²ψ`, then by `D_LCFS`
(lower is better). The development board currently separates first and second place by 0.0001, so
this needed stating before the Final phase rather than after it.

## 4. Tooling and authorship

AI coding assistants, including agentic ones, may be used for any part of an entry, provided the
workflow is disclosed in the methods report and the entrant takes responsibility for the submission.
For award candidates the reproduction package above is required regardless of how the code was
written. What remains prohibited is unchanged: submitting predictions you did not produce.

## 5. Documentation corrections

Several inconsistencies were reported and are being fixed:

- The Codabench **Overview** tab still showed the retired Challenge 2 gate (`R²ψ > 0.6`). The
  correct gate is the **full DIII-D composite `S ≥ 0.85`**, as the Rules and Evaluation tabs say and
  as the scorer enforces.
- The Hugging Face dataset card and the starter README still link competition **#17456**. The live
  competition is **#17739**.
- The starter README's summary units table is pre-v1.1.0 and says amperes. The authoritative
  statement is the Data tab and dataset card: shaping-coil currents are **kA·turn** (DIII-D
  `F1A`–`F9B`, MAST `p{2-6}{l,u}`); plasma current and DIII-D `ECOILA`/`bcoil`, MAST
  `sol`/`tf`/`efps` are **kA**. There is no `coil_turns` column and none is needed.
- **What decides the winners.** The Final phase description in the competition configuration says
  the best *Development* entry forms the basis for final ranking. That is wrong and is being
  corrected: the **Final phase, scored on the blind private fold, decides the winners**, as the
  Overview tab states.
- The site glossary describes the Consistency term as covering "eight ψ-derived scalars". It is
  **seven**: `R_axis`, `Z_axis`, `kappa`, `tri_top`, `tri_bot`, `volume`, `li`.
- `MODELING_GUIDE.md` says in one place that `dsep` is scored on diverted frames. It is **not
  scored at all** — as the same document says elsewhere. `dsep` is context only, and the two
  machines' columns are different physical quantities.
- **MAST solenoid scale.** `sol_current` reads exactly **2×** the
  FAIR-MAST level-2 `pf_active` solenoid current, and we have not resolved which convention is
  correct. If your forward model uses the solenoid quantitatively, treat the factor as an open
  question. `tf_current` and DIII-D `bcoil` have no poloidal rectangle (toroidal field coils), and
  `efps_current` is not a coil — it is a feed current at the P2 link board.

## With thanks

Every change above came from a participant telling us something we had not seen. Named with thanks,
and with apologies to anyone we have missed:

- **Vasily Gnuchev** — that the per-submission detailed results listed real machine shot numbers,
  and that the MAST fold is content-matchable against the public archive. Reported privately, with
  evidence, and not exploited.
- **Guillaume Pourcel** — that Challenge 1 is saturated at the top of the development board, and
  for asking how `G_ratio` is bounded, which is why the ceiling is now written down.
- **Jingyuan Gao** (starter issue #8) and **zhouyi-xiaoxiao** (issue #9) — for pressing on where
  exactly the zero-shot boundary falls, precisely enough that it could not be answered informally.
- **Team maximml** — for asking about the scope of "external public archives" before relying on
  their reading of it, and disclosing their own experiments unprompted.
- **pshishkin** — for the Challenge 2 gate analysis (issue #7) and the earlier data errata
  (issues #5 and #6).
- **Romeo** — for the dead competition links and the stale gate text on the Overview tab.
- **Michael Smolkin** — for asking whether a training shot was genuinely independent of the starter
  examples, which is the right question to ask before trusting a validation result.
- Several teams who wrote by email on coil units, circuit mapping and geometry conventions, and who
  found a units table of ours that was wrong by a factor of 1000.

Questions on any of this are welcome in this thread. Where an answer changes what is permitted, it
will be posted here rather than only in email, so everyone works from the same text.

— The organizing team
