# Clarifications: Development feedback on MAST, which Final submission counts, honorable mentions, and CPU data generation

- Thread: https://www.codabench.org/forums/17429/2827/
- Started by: sophelio
- Date created: Oct 02, 2026
- Pinned: yes
- Posts: 2

---

## Post 1 — sophelio

*Oct. 2, 2026, 3:44 p.m.*

Dear participants,

Several of you asked the same four questions, by email and on the starter repo. The answers below
are now in the Rules tab and the website FAQ. Two of them were never specified before and now are:
which Final submission counts, including how Award #2 is computed when you nominate separately, and
which machine the honorable mentions are judged on. The other two state precisely where an existing
rule draws its line.

### 1. Using Development scores to choose between MAST variants

**Question.** Rule 3 bars MAST equilibrium reconstructions from being used to "validate or select"
any MAST-predicting component. The Development leaderboard's MAST scores are computed against MAST
reconstructions. So does keeping the MAST variant that scored better count as selection?

**Answer: no — this is permitted, with disclosure.** The scores and Detailed Results the
Development phase returns for your submissions are the competition's own feedback. Every team gets
the same feedback, at the same rate (5 submissions a day, 100 in total), and you may use it to
compare, tune and select MAST pipelines. Say in your methods report which MAST choices it informed.

**Where the line is.** Rule 3 bars MAST equilibrium data **you can score against yourself,
offline**: archive reconstructions, the three labelled demo shots in the starter kit, or anything
derived from them. Those give per-shot answers without limit and without spending a submission. The
platform's scores do not. This is also what the 23 September table already said ("your own previous
submissions and their public-fold scores" are permitted in the MAST path). The rule text now says it
in words.

**Why we are not hiding the MAST columns.** It was suggested, and we considered it. Hiding MAST S
alone would not work, because `G_ratio × DIII-D S = MAST S`. Challenge 2 would have to lose all its
feedback, for every team, in the last weeks of a phase where that feedback has been public for two
months. The Final phase is also scored on a **different, blind set of MAST shots**, so tuning to the
public fold earns nothing in the Final that does not generalize.

If your team held back from using the Development scores since 23 September because it read rule 3
more strictly: that reading was reasonable given the wording, and you are free to use the feedback
from now on. Record your earlier choices in the methods report as you would anything else.

### 2. Which Final submission counts

You have up to **three** Final submissions.

- **Default:** your most recent Final submission that **scored successfully** counts for both
  awards and for the honorable mentions. A submission that fails to score never displaces an
  earlier one that scored.
- **Nomination (optional):** you may instead nominate one scored Final submission for **Award #1**
  and a different one for **Award #2**. Email `fusionec26@sophelio.io` with your team name and the
  two submission IDs **before the Final phase closes (2026-10-26 00:00 UTC)**. Each nominated submission needs its own
  reproduction package reference, as every Final submission already does.
- **How Award #2 is computed.** `S_MAST` comes from the submission that counts for Award #2. The
  denominator `S_DIII-D` is your team's **best DIII-D composite across all its successfully scored
  Final submissions**, and the `S ≥ 0.85` gate applies to that best score. If you make only one
  Final submission, this is exactly the `G_ratio` the scorer reports. Otherwise we compute it from
  your submissions' scores.
- **Why the denominator is your best DIII-D.** Without it, nomination would invite a known trick:
  submit your best DIII-D model for Award #1, then nominate a second submission with a deliberately
  weaker DIII-D model, which shrinks the denominator and inflates `G_ratio` (the problem raised in
  starter issue #7). Dividing by your best DIII-D removes the incentive. A weaker DIII-D model, in
  any of your submissions, cannot raise your Award #2 score, so no separate rule against it is
  needed. You remain free to pair a DIII-D model with a different MAST pipeline in one
  submission.
- **Platform faults.** If a Final submission sticks in the queue or fails for a reason on our side,
  email us and we will remove it so the slot comes back. With three slots, that matters.

### 3. Honorable mentions

The four honorable mentions are best `R²ψ`, best `R²{q95,βN}`, best `Consistency`, and best
`D_LCFS` (lowest). Each is judged on the **MAST blind private fold**, using the submission that
counts for Award #2. They are compared at full precision; exact ties are shared. Mention winners are
checked against their reproduction package before they are announced, as award candidates are.

### 4. The 24 GPU-hour rebuild ceiling and CPU data generation (issue #10)

- **Only GPU work counts** toward the 24 GPU-hour ceiling.
- **CPU-only data generation** — for example synthetic equilibria from your own free-boundary
  solver, which rule 3 permits — must be declared in the package with its hardware and wall-clock,
  but it is not capped. GPU time spent generating data does count.
- You **may ship the generated data** with SHA256 checksums, and our rebuild may start from it,
  provided the package also contains the code, configuration and random seeds that generate it. We
  may regenerate part of it to confirm it matches.

### 5. How compliance is checked in the Final

Several of you asked this directly. Every Final submission carries a reproduction package reference
at submission time. For the top five entries per challenge, we retrain from the training
entrypoint, re-score the rebuilt model, check that the shipped weights reproduce the submitted
predictions, and read the per-machine data-provenance statement and the methods report. An
inaccurate provenance statement disqualifies an entry in its own right.

The Development leaderboard is not re-scored and remains a practice board. Awards are decided on the
Final phase.

Questions on any of this are welcome in this thread.

— The organizing team

---

## Post 2 — sophelio

*Oct. 5, 2026, 2:41 p.m.*

## Follow-up: probe submissions, the envelope mask, and rebuild tolerance

Participants have asked follow-up questions about point 1 above (using Development scores for
MAST). The first two answers below are now in rule 3.

**1. What "compare, tune and select" covers.** You may use the score each submitted candidate
receives to choose among the candidates you submitted, including several settings of one knob. What
you may not do is use submissions as measurements of the labels, for example a constant, shifted or
rescaled prediction submitted to learn something about the labels rather than to be judged. Label
statistics obtained that way count as MAST equilibrium data under rule 3. So does anything fitted to
them: a bias or scale calibration, ensemble weights, or post-processing constants found by computing
scores offline. None of it may be part of a Final MAST pipeline.

Why the line sits there: the public and private MAST folds are stratified halves of one population,
so a mean or variance read from one holds for the other. Calibrating to it is calibrating against
MAST EFIT reconstructions, with the leaderboard as the go-between. Choosing among submitted variants
is different: it only tells you which candidate did better.

This is not retroactive, and Development entries are not judged. What matters is that no Final
submission or reproduction package contains such constants. If you have probe-derived numbers,
leave them out of your Final pipeline and describe what you did in your methods report.

**2. The envelope mask in `fusion_scoring/masks/`.** `mast_envelope.npz` is the union of every
ground-truth LCFS in the MAST corpus, dilated by two cells. That makes it an aggregate of MAST
labels, which rule 3 covers. It ships because the official metric uses it: when the scorer derives
the boundary and scalars from your predicted maps, it bounds the LCFS search with this mask.
Running that official derivation on your own predictions is fine, since it is what the platform
does when it scores you. Using the mask as a model input, feature, loss mask or prior for MAST
predictions is not allowed. The MAST wall and limiter geometry is permitted and serves the same
purpose.

**3. A MAST problem does not cost Award #1.** Award #1 is judged on DIII-D alone. If a team's MAST
pipeline breaks rule 3, the team loses eligibility for Award #2 and the honorable mentions, not for
Award #1. The exception: a provenance statement or methods report that misstated what was used
disqualifies the team from every award.

**4. Rebuild tolerance.** Before the Final opens on 19 October, we will post how closely a rebuilt
model must reproduce its submitted score.

— The organizing team
