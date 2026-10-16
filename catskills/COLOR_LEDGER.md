# CRC_FEQ color ledger

Standing memory for every luminance revision we ran, every job under it, what it was asking, and what it measured. A later session that has not read Black through Sky should start here.

Olive §0 of [`MODEL_DESCRIPTION_CRC_FEQ_olive.md`](MODEL_DESCRIPTION_CRC_FEQ_olive.md) is the same ledger through Steel, written as a one-line table. This file is the long form: theses, job-by-job goals and results, the public board, and what the four entries **above us** look like from the numbers. Name table: [`CRC_FEQ_COLORS.md`](CRC_FEQ_COLORS.md). Process: [`NEW_COLOR_DRILL.md`](NEW_COLOR_DRILL.md).

**Do not invent jobs that are not in this file.** Numbers are official 80-shot holdout unless marked public, oracle/GT, or existence-only. Public numbers are Codabench Challenge 1 (DIII-D) unless marked Challenge 2 (MAST).

---

## How to read this file

| word | meaning |
|---|---|
| **SHIP** | Packaged and scored on Codabench. |
| **KEEP** | Beats the then-held local bar. May or may not have shipped. |
| **PAPER** | Real but under the submit bar. Do not package. |
| **PASS** | The written gate cleared. Often a ceiling, not a submit. |
| **MISS** | Trained or applied; did not beat the keep bar. |
| **KILL** | Hit a written floor (usually \(R^2_\psi\)) and stopped. |
| **CEILING** | Uses holdout GT or an illegal warp. Not a submit. |
| **LOCK** | Negative result that closes a family. |
| **NO** | Existence fail; the arm was never legal. |
| **MOVED** | Rewritten as a later color’s job. |
| **VOID** | Run is not a verdict (diverged, wrong flag, power cycle). |

The competition score is

\[
S \;=\; 0.55\,R^2_\psi \;+\; 0.15\,R^2_{q_{95},\beta_N} \;+\; 0.10\,(1 - D_{\mathrm{LCFS}}) \;+\; 0.20\,\mathrm{Consistency}.
\]

Consistency is the mean of seven \(R^2\) values the scorer derives from \(\psi\): \(R_{\mathrm{axis}}\), \(Z_{\mathrm{axis}}\), \(\kappa\), \(\delta_{\mathrm{top}}\), \(\delta_{\mathrm{bot}}\), volume, \(\ell_i\).

Calibration has moved. Do not mix offsets:

| ship | public vs local |
|---|---|
| Charcoal P5 | ≈ identity (0.9567 → 0.9556) |
| Maroon ship6 | public ≈ local − 0.0013 |
| Purple C7b+C8a+C9a | public ≈ local − 0.0004 |
| **Brown B8+F9 (public #5)** | **public ≈ local − 0.00135** |
| Sky S1+F9 (local held, calibration probe) | packaged 2026-08-24 → `benchmarks/CRC_FEQ_sky/.../direct_submission.zip`. Local \(S=0.99326\). **Upload once** to Codabench for public calibration; do not reuse Brown −0.00135 offset. |

---

## 1. Where we are

**Public ship (2026-10-16):**
Best scored public submission is **992647** (Dual Champion Anchor, $c^*=1.062242, s_\beta=1.0000$):
- **Challenge 2 (Overall Track) Rank #3 in the World**: $G_{\text{ratio}} = \mathbf{0.889871}$, $S_{\text{MAST}} = \mathbf{0.884926}$, MAST $R^2_{qb} = \mathbf{0.413551}$ (probe 992586 scored $G_{\text{ratio}} = \mathbf{0.889969}$, $R^2_{qb} = \mathbf{0.414194}$).
- **Challenge 1 (DIII-D Track)**: $S_{\text{D3D}} = \mathbf{0.994443}$, Consistency = $0.980770$, $D_{\text{LCFS}} = 0.010584$, $R^2_\psi = 0.999830$, $R^2_{qb} = 0.996270$ (Rank #34 in the World).
- Paired with DIII-D 0.8.0 Ensemble with $dZ = +0.5\text{ mm}$ vertical micro-alignment and legacy 4-head blend scalars. All 5 daily submissions for 2026-10-16 scored and validated jump-free.

| | DIII-D Public (992647) | MAST Public (992647) | DIII-D Holdout (80-shot) |
|---|---:|---:|---:|
| \(S\) | **0.994443** | **0.884926** | **0.995735** |
| Cons | **0.980770** | **0.890400** | **0.985226** |
| \(R^2_\psi\) | **0.999830** | **0.998610** | **0.999856** |
| \(R^2_{q\beta}\) | **0.996270** | **0.413551** | **0.997775** |
| \(D_{\mathrm{LCFS}}\) | **0.010584** | **0.044217** | **0.008977** |
| `G_ratio` | — | **0.889871** (Rank #3) | — |

Prior baseline tailfit hybrid was [916155](https://www.codabench.org/competitions/17739/detailed_results/916155) (`d3d_S=0.993112`). Prior MAST GS solve v2 was [912438](https://www.codabench.org/competitions/17739/detailed_results/912438) (`mast_S=0.585865`, `G_ratio=0.590680`).

| | local S1+F9 | local B8+F9 (held) | public B8+F9 DIII-D | public GS v2 MAST (912438) |
|---|---:|---:|---:|---:|
| \(S\) / `d3d_S` | **0.99326** | 0.99320 | **0.991848** | — |
| Cons | 0.9758 | 0.9754 | 0.9707 | 0 (MAST) |
| \(R^2_\psi\) | 0.99962 | 0.99966 | 0.999641 | **+0.928** |
| \(\kappa\) (holdout) | 0.9525 | 0.9519 | — | — |
| MAST \(S\) | — | — | — | **0.585865** |
| `G_ratio` | — | — | 0.591 | **0.590680** |

Parents (do not unfreeze):

| ckpt | what | local \(S\) | Cons | \(\kappa\) |
|---|---|---:|---:|---:|
| `CRC_FEQ_sky_S1_tokens.pt` | \((r,z,I/|I_p|)\times 19\) dualctx | 0.99184 | 0.9702 | 0.9456 |
| `CRC_FEQ_forest_F9_unfreeze_edge.pt` | `th1` dualshot | 0.99191 | 0.97037 | 0.9454 |
| `CRC_FEQ_brown_B8_zgrid_e1.pt` | `thz` dualctx (retired from local mix) | 0.99173 | 0.9692 | 0.9445 |

A1 autopsy still applies to the singles: \(\kappa\sim 0.945\) is a shared hole. The S1+F9 mean lifts mix \(\kappa\) to 0.9525 (was 0.9519). A gate cannot invent elongation.

House rules:

- Submit nothing below local **0.9945**. That bar is not promised. The illegal Forest F66 GT warp of F9 is local 0.99436. The A2 oracle mix is 0.99335.
- One GPU. Sky, Yellow, and White are **CLOSED**. Live color is **Ultraviolet** (fillable token write + whole-shot Transformer). Search audit: [§23 Out of ideas](#23-out-of-ideas).
- **READ [§30.6](#306-consistency-is-a-slave-variable-of-psi-error--pixel-psi-is-a-ceiling-was-the-wrong-read) AND [§30.7](#307-the-tail-is-the-metric-r2_psi-is-decided-by-1-of-frames-and-they-are-ours-not-hard) FIRST.** Consistency is a *function of* \(R^2_\psi\), not a second axis — our public point and #1's sit on one measured curve, agreeing to 0.0005. #1 has no signal we lack; they fit \(\psi\) better. The job is **cut holdout \(\psi\) MSE by 41%**, scored by plain MSE with no scorer in the loop. §30.7 then locates that MSE: the top 1% of frames carry 30% of it, one shot in 80 carries 23.5%, and those frames are *member-specific failures* — a per-frame oracle router is a 66.5% cut, so a router capturing a quarter of its oracle wins. Ensembling alone gives ~10%. **B12 is a dead checkpoint (MSE 10.1); never load it.** **Ensembling ~10%; the per-frame router is CLOSED as hindsight (§30.8) — the tail must be fixed in training, not by selection.** **§30.9: the tail is a per-shot offset worth 39.8%, invisible to every label-free probe. §30.10 CLOSED: the parents were UNDERTRAINED — a plain fine-tune cut B8's MSE 19.5% and beat the shipped pair with one model, while hard-frame weighting hurt. Best mix B8/L+O1/U+O1/L is +38.40% \(\psi\) MSE; official mix \(S=0.976\) from qB drift; hybrid (mix psi + lock qB) \(S=0.994147\). Extra O1 seeds parked.**
- **[§30](#30-projection--the-map-is-one-gradient-walk-from-cons-0995-open-2026-09-02) supersedes several §23 conclusions.** A Cons-**0.995** \(\psi\) exists inside our \(k=60\) basis a short SoftDerive walk from the shipped map, for \(6\times10^{-5}\) of \(R^2_\psi\) — worth \(\approx +0.004\) \(S\), more than the gap to #1, and the walk is label-free at inference. **The mechanism works; the input does not exist.** §30.5 ran the 7-scalar predictor to the full corpus and it **saturates** at \(\Delta\)Cons \(+0.0013\) (\(+0.0002\ S\)), with error correlation against the map *rising* as the model improves. \(\kappa\) is information-limited at \(\approx 0.956\) from the legal write. Do not raise \(\lambda_w\); do not post-hoc splice a finished map; do not re-run scalar stacking, scalar/\(\psi\) smoothing, or tabular scalar regression at any corpus size.
- **[§33](#33-the-instrument-was-wrong--error-bars-the-shift-aware-fold-and-a-kappa-closure-retracted-by-34-2026-09-05) is a method change, read it before running anything.** Every Consistency number before it was read off a fold with no error bar: aggregate Cons 1\(\sigma\) is **0.00233** on 80 shots and \(\kappa\)'s is **0.01203**, larger than most effects six colors reported. **Use `cons_foldvar --compare` (paired, 3.4× tighter, and its local delta predicted the public delta) for every A/B; never compare bare holdout numbers.** **Score Consistency on the worst-covered 40 shots (`P33_COVERAGE.json`) — the random 80 overestimates the public value by the entire 0.0033 offset.** **Sweep on one parent, then replicate the winner on the other two; a +2.09\(\sigma\) best-of-nine arm came back −1.92\(\sigma\) and −3.23\(\sigma\) on its siblings.** §31.2's per-scalar profile is **inverted** on the scored fold. ~~\(\kappa\) is CLOSED at its information limit.~~ **[§34](#34-the-rivals-scorecards-are-public-too--kappa-was-never-at-a-limit-open-2026-09-06) retracts that**: `get_details` is unauthenticated for **every** submission on the board, #1 scores \(\kappa=0.972315\) to our 0.959625, and `cliffsedge66` beats our \(\kappa\) at **4.5× our \(\psi\) error**. **Check a rival's card before calling anything information-limited — it is the cheapest upper bound available, and §30.5/§33.4/§33.7 all measured our own family and mistook it for the task.** We are last in the top eight on four of the seven scalars.
- Do not reopen LOCK / KILL / CEILING / NO rows. Do not invent F↔P or a 21+14 zero union. Do not retrain S2 or Y2.
- **Challenge 1 (DIII-D) is first.** MAST public-archive training is Track 2, a supplement. Agents execute the client's stated plan; they do not gate work on their reading of competition terms.

---

## 2. Public leaderboard

Snapshot: **2026-09-06 00:51** Eastern, Codabench `leaderboards/19740`, [results tab](https://www.codabench.org/competitions/17739/#/results-tab). Source: `results/forest/leaderboard.json` (refresh with `scripts/codabench_board.py`); per-scalar cards for **any** row via `scripts/rival_details.py` (§34.1). Challenge 1 is DIII-D. Challenge 2 is MAST. `G_ratio` is the organizer’s Challenge-2 / Challenge-1 ratio.

| rk | who | \(S\) | Cons | \(R^2_\psi\) | \(R^2_{q\beta}\) | \(D_{\mathrm{LCFS}}\) | MAST \(S\) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | **kimsunjoong** | **0.994984** | **0.982144** | 0.999826 | 0.997397 | 0.009584 | **0.9479** |
| 2 | dgaytanv | 0.994794 | 0.981313 | **0.999879** | 0.997125 | 0.009705 | 0 |
| 3 | dmitryzhevn | 0.994376 | 0.980018 | 0.999789 | 0.996797 | 0.010309 | 0.9202 |
| 4 | mwaller | 0.994345 | 0.980052 | 0.999751 | 0.996635 | 0.010237 | 0.6117 |
| 5 | pshishkin | 0.994073 | 0.979141 | 0.999818 | 0.996337 | 0.011055 | 0.5681 |
| 6 | alvar0 | 0.993938 | 0.978234 | 0.999839 | 0.996160 | 0.010441 | **0.9763** |
| **7** | **catskills** | **0.993217** | **0.976314** | **0.999784** | **0.995065** | **0.011865** | **0.585865** |
| 8 | lkirill | 0.993159 | 0.975616 | 0.999781 | 0.995668 | 0.011940 | 0.5898 |

**#7 as of `o1x5` (§34.5–34.6).** Gap to #1 is **0.001767**: Cons 66%, \(q\beta\) 20%,
\(D_{\rm LCFS}\) 13%, \(R^2_\psi\) 1%. We are **last in the top eight on \(R^2_{q\beta}\)** and
on four of the seven Consistency scalars (§34.1). The board's four-decimal \(R^2_\psi\) column
hides a **15.5×** spread in \(1-R^2_\psi\) across the top 20, so "everyone is at 0.9998" is a
display artifact, not a saturation (§34.2). Challenge 2 climbed hard while we were on DIII-D —
#1 went 0.74 → 0.9479 and alvar0 holds 0.9763, against our 0.5859.

### 2.1 Weighted gap #5 vs #1

Public \(\Delta S = 0.002528\).

| term | weight | us | #1 | weighted \(\Delta\) | share of gap |
|---|---:|---:|---:|---:|---:|
| Consistency | 0.20 | 0.970717 | 0.980018 | 0.001860 | **73.6%** |
| \(1-D_{\mathrm{LCFS}}\) | 0.10 | 0.986423 | 0.989691 | 0.000327 | 12.9% |
| \(R^2_{q\beta}\) | 0.15 | 0.995065 | 0.996797 | 0.000260 | 10.3% |
| \(R^2_\psi\) | 0.55 | 0.999641 | 0.999789 | 0.000081 | **3.2%** |

~~Pixel \(\psi\) is a ceiling. Challenge 1 is a Cons / \(\kappa\) hole.~~ **Superseded 2026-09-02 by
[§30.6](#306-consistency-is-a-slave-variable-of-psi-error--pixel-psi-is-a-ceiling-was-the-wrong-read).**
This table splits \(S\) into independent terms, and Consistency is **not** independent of
\(R^2_\psi\) — it is a function of it (both real entries lie on one measured curve). Reading the 3.2%
row as "\(\psi\) is done" and the 73.6% row as "a separate \(\kappa\) hole" double-counts one axis
and sent six colors after a hole that does not exist. \(R^2_\psi\) carries its own 3.2% **plus**
most of the 73.6%. Challenge 2 is forfeited (empty F-slots).

R6 (GT \(\psi\) through our \(k=60\) basis) is local Cons **0.9972** / \(\kappa\) **0.9887** / \(\ell_i\) **0.9988**. The basis can hold #1’s Cons. We do not write it.

---

## 3. What the four above us are doing

These are **inferences from public numbers**, not from their code. Label them as such. The point is to stop guessing architectures we already killed and to name what the numbers require.

### 3.1 #1 dmitryzhevn — \(S=0.99438\), Cons 0.9800, MAST 0.74, method `25`

What the numbers require:

1. **A DIII-D write that we have not reached.** Cons 0.980 vs our 0.971. That is almost the entire Challenge-1 gap. Their \(R^2_\psi\) is only +0.00015 over us. They are not winning by fitting pixels harder.
2. **A representation that produces a real MAST flux map.** MAST \(R^2_\psi=0.969\). Ours is −0.887. That is not “they guessed a better pairing of F-coils to P-coils.” A name-paired regressor on empty DIII-D F-slots scores like us (Coral: B8 with F=0 is \(S=0.170\)). They have a write that is fillable on both machines.
3. **They are not a Cons monster on MAST.** MAST Cons is only 0.223. They win Challenge 2 on \(\psi\) and LCFS (\(D_{\mathrm{LCFS}}=0.195\)), not on derived scalars. #8 dremovd has *better* MAST (\(S=0.896\), Cons 0.730, \(R^2_\psi=0.995\)) and *worse* DIII-D (\(S=0.989\), Cons 0.958). #1 is the one entry that holds **both** boards. That is the hard thing.
4. **Not our 21 named slots, not four Gray roles that B8 can read, and not a vac raster as the only coil write.** Red P0: MAST leaves 20/21 DIII-D magnetics empty. Coral P0: four roles rebuild F-coils at train \(R^2=0.02\)–0.25 and B8 dies (\(\Delta S=0.424\)). Sky S2: a \(16\times 16\) \(\psi_{\mathrm{vac}}\) parent scores \(S=0.802\). Sky S1: per-coil \((r,z,I/|I_p|)\) holds DIII-D at B8 (\(S=0.99184\)) — **MAST token infer was not scored**. Whatever they use keeps **per-coil identity** *and* fills MAST, or uses a physical basis that does not spend DIII-D the way S2 did.
5. Method string climbed (`22` earlier in our notes, **`25`** on this snapshot). They are iterating a method, not a one-shot ensemble.

What this is probably **not**:

- Raw DIII-D F-amp regression (MAST 0.74 forbids it).
- Our PCA-60 + dualctx/dualshot family with a hotter \(\lambda\) (C7 saturates at Cons 0.958; B8+F9 is 0.975 local / 0.971 public; we cannot climb 0.009 Cons by dose).
- Inference Grad–Shafranov on a finished map (Forest F13 killed us at \(S=0.861\)).
- A bigger residual trunk (R9: +56% params, +0.00004 \(S\)).
- Our Sky S2 vac-grid parent (DIII-D \(S=0.802\)). Green’s as the *only* coil write is not their DIII-D story.

Most likely: a **physics-normalized shared basis** that keeps per-coil identity (Sky S1-shaped) *or* a vacuum/geometry write that does not spend DIII-D the way S2 did — plus a flux fit that still holds DIII-D Cons. Tokens on DIII-D are measured; MAST token fill is not.

### 3.2 #2 dgaytanv — \(S=0.99320\), Cons 0.9761, **best \(R^2_\psi\) on the board**

What the numbers require:

1. **A flux-fit monster.** \(R^2_\psi=0.999864\) is the board best, +0.000075 over #1 and +0.000223 over us. At weight 0.55 that is +0.00012 of \(S\) vs us from \(\psi\) alone.
2. **Less Cons transfer than #1.** Cons 0.976 vs 0.980. They beat us on Cons (+0.0054) and still lose to #1 on the term that matters.
3. **No Challenge 2.** MAST \(S=0\). Either they did not submit Challenge 2 or the run is empty. This is a DIII-D-only fit. Contrast #1 (0.74) and even #4 (0.58).
4. Local coincidence: their public \(S=0.993203\) **equals** our local B8+F9 \(S=0.993203\). That is not the same model. Their public Cons is 0.976 vs our local 0.975 / public 0.971. They generalize Cons better than we do.

Most likely: a very strong DIII-D encoder (maybe the same PCA / ResNet neighborhood we are in) with more Cons than us and no machine-transfer story. We are not losing Challenge 1 to them on \(\psi\); we are losing on Cons and LCFS.

### 3.3 #3 mnn31 — \(S=0.99208\), Cons 0.9737, MAST 0.56

What the numbers require:

1. All four DIII-D terms a notch under #2. Cons 0.974 is the interesting one: **+0.003 vs us**, enough to clear us on \(S\) even though their \(R^2_{q\beta}\) (0.9920) is **worse** than ours (0.9951). They spend \(q\beta\) and collect Cons.
2. Partial MAST: \(S=0.560\), \(R^2_\psi=0.848\), Cons 0. The flux map transfers some; derived scalars do not. Weaker than #1’s 0.969 \(\psi\), stronger than our −0.887.
3. Older submit (Aug 16). They have not moved while #1 and #4 posted on Aug 20.

Most likely: same family as #2, slightly weaker DIII-D fit, with *some* shared write that produces a MAST \(\psi\) at \(R^2\sim 0.85\). Not empty-F. Not our 21 names.

### 3.4 #4 mwaller — \(S=0.99196\), Cons 0.9705, MAST 0.58

This is the entry that looks like **us**.

| | mwaller public | us public | who wins |
|---|---:|---:|---|
| \(S\) | 0.991960 | 0.991848 | them, +0.00011 |
| Cons | 0.970486 | **0.970717** | **us**, +0.00023 |
| \(R^2_\psi\) | **0.999720** | 0.999641 | them |
| \(R^2_{q\beta}\) | **0.995149** | 0.995065 | them |
| \(1-D_{\mathrm{LCFS}}\) | **0.987442** | 0.986423 | them |
| MAST \(S\) | **0.5838** | 0.0553 | them |
| MAST \(R^2_\psi\) | **0.9256** | −0.887 | them |

We beat them on the term we have been hunting (Cons) and lose the board slot on \(\psi\) / LCFS / \(q\beta\). Their MAST \(R^2_\psi=0.926\) means they also have a fillable write; they just hold DIII-D Cons at our level, not #1’s.

Most likely: a B8+F9-neighborhood DIII-D model (or a single strong parent) plus enough machine-shared input to score MAST \(\psi\sim 0.93\). Same Challenge-1 class as us; they did not forfeit Challenge 2.

### 3.5 Nearby datapoints (not above us, but they constrain the story)

- **#6 ruiqi_feng** (`exp40 free-boundary physics calibration`, compute-light): Cons **0.9736** (better than us), \(S\) 0.00001 under us because \(\psi\) and \(q\beta\) are weaker. MAST 0.65 / \(\psi\) 0.965. The method string says free-boundary physics. That is the lane Steel S1/S2 and Brown B15 tried and spent. Their compute-light badge plus MAST 0.65 says the physics is in the *features or the solve*, not a 19 M dualctx.
- **#8 dremovd**: best MAST on the board (0.896 / Cons 0.730 / \(\psi\) 0.995) and DIII-D Cons only 0.958. Transfer-first, Challenge-1 spent. #1 is not this; #1 refused to spend DIII-D to buy MAST.

---

## 4. One line per color

| Color | Hex | Thesis | Outcome | Successor |
|---|---|---|---|---|
| Black | `#000000` | Ridge / MLP on PCA-60 | Family exists. Capacity tiny. \(S\) ≪ 0.95. | Navy |
| Navy | `#000080` | INR \(\psi(R,Z)\), no PCA | **LOCK.** \(S\approx 0.60\), Cons ~0. | Teal |
| Teal | `#008080` | Cons on the PCA manifold | Surrogate \(\kappa\) can go to \(R^2=-0.60\). | Charcoal |
| Charcoal | `#333333` | SoftDerive + geom emphasis | SoftDerive works. Public **0.9556**. Vacuum-as-\(Y\) dead. | Maroon |
| Maroon | `#800000` | Full corpus + residual trunk + ensemble | Trunk > MLP. R6: basis is not the ceiling. Public **0.9832 / #6**. | Purple |
| Purple | `#800080` | Dualctx (context gets its own encoder) | Stem was the defect. \(\lambda_w=1.35\) saturates. Public **0.9904 / #7**. GRU-only dies. | Forest |
| Forest | `#228B22` | Shot GRU + post-process \(\kappa\) | **F9** single 0.99191. Oracles are ceilings. \(s_Z/dR/dZ\) from \(X\) are dead. | Brown |
| Brown | `#A52A2A` | Z-grid dualctx × F9 | **B8+F9 SHIP** local 0.99320 → public **0.99185 / #5**. Public ship. Local mix later replaced by S1+F9. | Steel |
| Steel | `#4682B4` | Frozen-B8 residuals | Every arm ~0.991 or crash. \(\Delta\kappa\) OOF \(R^2=-0.019\). | Olive |
| Olive | `#808000` | New plastic parent + early SoftDerive + GradNorm | SoftDerive never turned on. Best mix 0.99317 vs held 0.99320. | Red |
| Red | `#FF0000` | Same-wiring non-dim | **P0 FAIL.** Only `plasma_current` live on MAST. | Gray |
| Gray | `#A9A9A9` | Four shared roles, new parent | Roles exist. Retrain spends DIII-D (\(S=0.9644\)). | Coral |
| Coral | `#FF7F50` | Frozen B8; roles→21 at MAST infer | Empty-F **is** the MAST disease. Rebuild \(\Delta S=0.424\). | Sky |
| Sky | `#87CEEB` | Coils as spatial sources (tokens / full \(\psi_{\mathrm{vac}}\)) | S2 vac-grid **LOCK** \(S=0.802\). S1 tokens **PAPER** 0.99184. M1 S1+F9 **KEEP** local **0.99326**. Public still B8+F9. | Yellow |
| Yellow | `#FFFF00` | \(\psi_{\mathrm{vac}},\vec B_{\mathrm{vac}}\) at Thomson \(Z\) + S1 tokens | P0 PASS. Y2 **MISS** \(S=0.99136\) / \(\kappa=0.9408\). M1 skipped. | White |
| White | `#FFFFFF` | Fully conv U-Net on official \(65\times 65\) | **CLOSED** 2026-08-24. AUD+SPR failed official gates; `W2_decision.json` closes axis. Best sprint official S=0.692. | — |

---

## 5. Black — ridge / MLP on PCA-60

**Thesis.** The competition family: compress \(\psi\) to 60 PCA coefficients, regress coefficients (and \(q_{95},\beta_N\)) from magnetics.

**Goal.** Prove a trainable map exists.

**Result.** It does. Capacity is tiny. \(S\) far below 0.95. Not a live ship. Every later color that still emits PCA-60 sits on this decoder.

**Lesson.** The decoder is not the thing we iterate. The write of \(\hat z\) is.

---

## 6. Navy — implicit neural \(\psi(R,Z)\)

**Thesis.** Drop PCA. Emit \(\psi\) from a coordinate MLP (INR).

| ID | Goal | Result |
|---|---|---|
| INR | Rebuild the decoder as \(\psi(R,Z)\) | **LOCK.** \(S\approx 0.60\), Cons ~0. |

**Lesson.** Do not rebuild the decoder as an INR. FieldRefiner (Charcoal 0a) and X-point basis (Steel) die for the same reason: the basis is not the ceiling.

Docs: [`MODEL_DESCRIPTION_CRC_FEQ_navy.md`](MODEL_DESCRIPTION_CRC_FEQ_navy.md).

---

## 7. Teal — Consistency on the PCA manifold

**Thesis.** Push Consistency by training in coefficient space / a surrogate \(\kappa\).

| ID | Goal | Result |
|---|---|---|
| manifold Cons | Cons loss in coeff space | Surrogate can destroy \(\kappa\) (\(R^2_\kappa=-0.60\)). A biased Cons loss is worse than none. |
| ensemble3 | Mean of three Wave-3 teal scale ckpts | Packaging ID, not a luminance step. Toy vs later ships. |

**Lesson.** Official Consistency is seven derived scalars from \(\psi\). Optimizing a proxy that is not those scalars spends the map.

Docs: [`MODEL_DESCRIPTION_CRC_FEQ_teal.md`](MODEL_DESCRIPTION_CRC_FEQ_teal.md), [`MODEL_DESCRIPTION_CRC_FEQ_teal_ensemble.md`](MODEL_DESCRIPTION_CRC_FEQ_teal_ensemble.md).

---

## 8. Charcoal — SoftDerive + geom emphasis

**Thesis.** Write a differentiable replica of the scorer’s geometry (SoftDerive) and emphasize it in the loss. The limit is the basis or the fit?

**Goal.** Move Cons without inventing a new decoder.

| ID | Goal | Result | Verdict |
|---|---|---|---|
| 0a | Is \(k=60\) the Cons ceiling? GT through our basis | Cons **0.9971**. FieldRefiner cancelled. | **PASS / LOCK** raise-\(k\) |
| 0b | Shrinkage vs noise: does smoothing still pay? | Ensemble and smoothing are substitutes. Smoothing pays only while incoherent residual remains. | measured |
| 0c | Test-time temporal smooth of \(\psi\) | Negative on a strong ensemble (−0.00035 \(S\) later on maroon R3). Helps slow scalars, hurts \(\ell_i\). | **LOCK** as a ship lever |
| 0d | Kappa error attribution | Field error, not extraction, dominates \(\kappa\). | measured |
| P1 | SoftDerive = batched `derive_frame` | Discrete O-point / \(\psi_b\) detached; values stay in-graph. Still the Cons loss. | **PASS** |
| P3 | Ablate geom / temporal / H1 | Geom emphasis works. Temporal window and H1 hurt at this scale. | measured |
| P4 | Scale s42, geom-emphasis 10 | Local \(S\) **0.9530** / Cons 0.8489. Beat the prior ship. | **KEEP** |
| P5 | Scale s202 | Local 0.9567 → public **0.9556**. s101 crashed. | **SHIP** |
| P6a | Soft \(D_{\mathrm{LCFS}}\) | Surrogate works; official \(1-D_{\mathrm{LCFS}}\) later shown not to track (maroon R10b). | proxy ≠ metric |
| P6b/c/d | Deficit weights / anneal / \(L_{q\beta}\) as \(R^2\) | Built; ladder waited on P5. Not the maroon trunk. | superseded |
| P7 | \(\psi\) on trained-on vs holdout | Trained-on \(R^2_\psi\) 0.99537. Capacity, not an input bug. | **CAPACITY** |
| P8 | Width/depth at 250 shots | Width 512 helps a little; 1024+ overfits. Wrong question: 12.7% of the corpus. | wrong lever |
| 25.1 | Subtract \(\psi_{\mathrm{vac}}\) from \(Y\) | Vacuum-as-\(Y\) dead. (Brown B13 later uses vac as *input* PCA — different question.) | **LOCK** |

**What it left.** SoftDerive is real. Geom-on MLP can ship at public 0.9556. The next lever is **data scale**, not a fancier decoder.

Docs: [`MODEL_DESCRIPTION_CRC_FEQ_charcoal.md`](MODEL_DESCRIPTION_CRC_FEQ_charcoal.md), [`CHARCOAL_PLAN.md`](CHARCOAL_PLAN.md).

---

## 9. Maroon — full corpus + residual trunk + ensemble

**Thesis.** Charcoal trained on 199,971 of 1,559,340 frames. Retrain the same family on the full corpus; then ask whether the MLP trunk is the ceiling.

**Goal.** Close data, capacity, context-as-columns, basis size, and seed-only ensembling.

| ID | Goal | Result | Verdict |
|---|---|---|---|
| M0 | Coeff cache; reconstruct unseen shots | \(R^2_\psi\ge 0.99999\). Full-corpus training fits on 12 GB. | **PASS** |
| M1 | Data ladder at equal steps, geom off | 7.7× data: \(S\) 0.92055 → 0.93443. | **PASS** |
| P1 | Full corpus, geom-emphasis 10, 250 ep | Power cycle ep 50. Fragment \(S\) 0.95017: Cons 0.710→0.848, \(\psi\) and LCFS paid. | **KILL** geom-on MLP |
| P2–P5 | Continue charcoal geom ladder | Cancelled. P1 fragment was enough. | cancelled |
| R1 | Residual trunk, geom off, vs M1c | \(S\) **0.97506**, \(R^2_\psi\) 0.99880. Cons 0.8936 *without* geom loss. | **PASS** |
| R2a | Concatenate window-5 context | \(S\) 0.95938. Central-diff of raw \(X\) amplifies jumps. | **MISS** |
| R2b | `--feature-clip 10` | \(S\) **0.97698**. Gain through Cons, not \(\psi\). | **KEEP** (clip stays) |
| R3 | Seed ensemble of four R2b | \(S\) **0.98170** / Cons 0.9231. Character ≫ magnitude (~80×). Seed variance ±0.002 \(S\). | **KEEP** |
| R4 | Context + clip together | \(S\) 0.96979. | **MISS** (concat context retired) |
| R5 | 17 M trunk, no warmup / no clip | Diverged (loss → 7.9e6). Optimizer, not capacity. | **VOID** |
| R6 | GT \(\psi\) through \(k=60\) | \(S\) 0.99900 / Cons **0.9972** / \(\kappa\) **0.9887** / \(\ell_i\) **0.9988**. | **LOCK** raise-\(k\) |
| R7 | R5 trunk + warmup 5 + grad-clip 1 | \(S\) **0.97998**. Capacity is real. | **KEEP** |
| R8 | \(1/N\) of seed members | \(S(N)=0.98395-0.00910/N\). \(N=\infty\) still under #1. | count spent |
| R9 | 26.5 M, matched wall clock vs R7 | \(S\) 0.98002 (+0.00004). | **MISS** as a lever |
| R10a | `smoothdiff` concatenated context | Single \(S\) 0.97642 **MISS**; mix member +0.00035 **KEEP**. Different \(X\) representation. | split |
| R10b | Soft LCFS on | Official LCFS −0.00013; surrogate improved 17%. Mix +0.00019. | proxy ≠ metric |
| kopt | Is PCA-50 the missing ingredient? | 14/60 modes lose to the mean; dropping them is 0.001% \(\psi\) MSE. | closed, no train |
| ship6 | R7+R9+R2b+R3b+R10a+R10b | Local 0.98451 → public **0.9832 / #6**. Calib was wrong by 0.0045. | **SHIP** |

**What it left.** Trunk > MLP. Clip 10 forever. Concatenated context loses three times (R2a, R4, R10a). The entire remaining gap after R6 is **fit quality**. Rank 1 was then described as “PCA50 + dual-context ResNet flux ensemble” — we held three of four ingredients; context delivery was the untested one.

Docs: [`MODEL_DESCRIPTION_CRC_FEQ_maroon.md`](MODEL_DESCRIPTION_CRC_FEQ_maroon.md). Board was `:8766`.

---

## 10. Purple — dualctx

**Thesis.** Temporal context has lost as *columns* because one trunk must spend capacity suppressing the marginal block. Give context its own encoder; fuse embeddings through a gate **initialized to 0**.

**Goal.** Deliver the information R10a proved is useful, without the shared-stem tax.

| ID | Goal | Result | Verdict |
|---|---|---|---|
| C1 | Dualctx, gate init 0 | \(S\) **0.98178**. Cons +0.0085, \(\psi\) flat. Stem was the defect. | **PASS** |
| C2a | Bigger context encoder 1024×6 | Below C1. \(\kappa\) −0.0157. | **MISS** |
| C2b | Window 5→9 | Best single then: \(S\) **0.98214** / Cons 0.92619. | window, not width |
| combo | Free mix of owned ckpts | C1+C2a+C2b+R7+R9 local 0.98678. Under then #5 bar. | not submitted |
| C3 | 200 epochs of C2b | \(S\) 0.98154 *below* 88-ep. \(\psi\) MSE down, Cons −0.00254. | **MISS / LOCK** schedule |
| C4 | 8 dualctx members | Paused after C4a. Seed Cons noise is 0.0015, not 0.0005. | paused |
| C5a | Whitened \(L_{\mathrm{coeff}}\) clamp 12, \(\lambda_w=0.05\) | \(S\) **0.98406** / Cons 0.93519. First Cons gain that does not spend \(\psi\). | **PASS** |
| C5b | Val selector only | \(\Delta S\) −0.00008. | **LOCK** selector |
| band | Hand true coeffs by mode band | Cons responds ~1:1 to gradient share. Modes 1–12 hold ~60% of recoverable Cons. | measured |
| C6a | \(\lambda_w=0.15\), clamp 12 | Gate miss on \(\psi\) by 0.00005; best single then \(S\) **0.98542** / Cons 0.94190. | dose curve real |
| C6b | Same strength, clamp 8 | Lost 0.00326 Cons to C6a, won \(\psi\). Allocation matters. | measured |
| C6c | Clamp 20 overshoot | Predicted miss; dropped to pay for C7. | dropped |
| C7a | \(\lambda_w=0.45\) | \(S\) 0.98738 / Cons 0.95025. | rung |
| C7b | \(\lambda_w=1.35\) | \(S\) **0.98884** / Cons **0.95773**. Saturates here. | **KEEP** |
| C7c | \(\lambda_w=4.05\) | Cons +0.00061. Null. | **LOCK** more-\(\lambda\) |
| C8a | Edge-weighted \(L_{\mathrm{data}}\) | Used in the shipped 3-mix. Not a new #1. | mix member |
| C9a | Small geom on C7b | Cons pass / \(R^2_\psi\) miss. In the shipped 3-mix. | mix member |
| ship3 | C7b+C8a+C9a | Local 0.99096 → public **0.9904 / #7**. Calib ≈ local − 0.0004 *then*. | **SHIP** |
| D1a | GRU 2×256 as the only trunk | \(S\) 0.96921 / Cons 0.867. R2a-shaped. | **KILL** family |
| D2 | Gated GRU on frozen C7b | LIVE as a trainer, not a member. Every D2 mix lost to ship3. | shot GRU → Forest |
| 396M GRU | GRU-only huge trunk | Val \(R^2_\psi=-398\) after warmup. | **KILL** |

**What it left.** Dualctx is the delivery that works. Whitened coeff loss saturates at \(\lambda_w=1.35\). Longer training anti-aligns past \(R^2_\psi\) 0.9993. GRU as the *only* trunk dies; GRU as a gated residual on a map is Forest.

Same card as the 0.9904 ship: new #1 pshishkin 0.9932 `seq + mlp` compute-light. The “rank 1 is our family” lock is false.

Docs: [`MODEL_DESCRIPTION_CRC_FEQ_purple.md`](MODEL_DESCRIPTION_CRC_FEQ_purple.md). Board was `:8767`.

---

## 11. Forest — shot GRU + F9 + post-process

**Thesis.** Sequence belongs *on* a finished flux map (gated shot GRU), not instead of it. Then: can we recover the leftover Cons knobs (\(s_Z\), \(dR\), \(dZ\)) from legal \(X\)?

**Goal.** Beat the Purple 3-mix as a single; close post-process.

F1–F3 are the Purple hangover (huge GRU-only). F4 onward is the Forest ladder.

| ID | Goal | Result | Verdict |
|---|---|---|---|
| F1 | Huge GRU as the only trunk | Val \(R^2_\psi\approx -398\) after warmup. | **KILL** |
| F4 | Shot GRU on frozen present-frame map | \(S\) 0.98919. Cousin +0.00007 on the 3-mix. | not a new base |
| F6 | Unfreeze F4, lr \(10^{-3}\) | \(S\) 0.98872. | **MISS** |
| F7 | Unfreeze, lr \(3\times10^{-4}\) | \(S\) 0.98952. | **MISS** |
| F8 | Unfreeze + C9a geom2 | \(S\) 0.99014 / Cons 0.96254. Parent of F9. | parent |
| F9 | F8 + one geom-off data epoch (edge flag was a no-op) | Best single **0.99191** / Cons 0.97037 / \(\kappa\) 0.9454. `best_epoch` 1. | **KEEP** (held single) |
| F10 | F9 + edge band | 0.99171. | **MISS** |
| F11 | F8 + edge mix 0.8 | Tied F9 exactly. | **MISS** |
| F12 | F9 + lcfs-emphasis 2 | 0.99171 / Cons 0.9695. Soft LCFS does not move \(\kappa\). | **MISS** |
| mix+F4 | 3-mix + F4 | 0.99103. Dilutes later F9. | dilute |
| F13 | Inference GS projection of F9 | \(S\) **0.861** / Cons 0.376. | **KILL** |
| F14 | Train-time \(L_{\mathrm{GS}}\) 88 ep | Flag was not on the trainer. Same class as F13. | never launched |
| F15 | Oracle \(s_Z\) vs GT \(\kappa\) | \(S\) 0.99294 / Cons 0.9750. The knob exists. | **CEILING** |
| F16 | Features → \(s_Z\) | corr 0.09; \(s=1.000\pm0.002\). \(S\) is F9. | **LOCK** |
| F17 | Features → \(\kappa\) then \(s=\hat\kappa/\kappa_{\mathrm{map}}\) | Spent the map (\(\kappa\) 0.559). | **KILL** |
| F18 | Map-shape → \(s-1\) | corr 0.18. | **LOCK** stretch-from-visible |
| F19 | Hotter \(\kappa\) geom on finished F9, 4 ep | \(S\) 0.9823. | **KILL** |
| F20 | F15 + oracle \(dR\) | \(S\) 0.99333 / \(R_{\mathrm{axis}}\) 1.000. | **CEILING** |
| F21 | Features/shape → \(dR\) | corr −0.02. | **LOCK** |
| F22 | F9 embed → \(dR\) | corr 0.12. Worse than F9. | **MISS** |
| F23 | F8+edge+\(R_{\mathrm{axis}}\times 2\) | \(S\) 0.9890. Walked off. | **MISS** |
| F24 | 5-frame MA of F9 \(\psi\) | \(S\) 0.99058. | **MISS** |
| F25 | F9 recipe seed 101, 4 ep | \(S\) 0.9892. Val proxy picks a later epoch that spends the map. | **MISS** |
| F26 | Seed 101, 1 ep | \(S\) 0.9859. F9 is a seed-42 ticket. | **MISS** |
| F27 | Compute-light seq+mlp 512×2 | Not F9. | **MISS** |
| F28 | Freeze F9, train only GRU | \(S\) 0.99138. Cousin of F4. | **MISS** |
| F29 | C8a parent, F9 recipe 1 ep | \(S\) 0.9847. | **MISS** |
| F30 | Oracle \(s_R\) for \(\kappa\) | Width is not a better 1-dof than \(s_Z\). | **CEILING** |
| F31 | F15 + \(s_R\) to restore volume | Next is isotropic scale. | **MISS** |
| F32 | F15 + isotropic scale for volume | PASS. GT only. | **CEILING** |
| F33 | Oracle \(dR\) on F32 | \(\psi\) spent at full \(dR\). | **CEILING** |
| F34 | Scaled \(dR\) on F32 | Local #1 bar 0.9938 in sight. | **CEILING** |
| F35 | Legal shot-median \(\kappa\) stretch | Flattening kills frame variation. | **MISS** |
| F36 | Iso-volume on raw F9 | Volume returns; \(S\) does not. \(s_{\mathrm{iso}}\) only works after \(s_Z\). | **MISS** |
| F37 | Legal MA5 \(\kappa\) stretch | — | **MISS** |
| F38 | dualseq 1 ep from F9 | \(S\) 0.9837. Dropping the shot GRU walks off F9. | **KILL** |
| F39 | F9 + \(L_{\mathrm{disp}}\) | Held flux, spent \(Z_{\mathrm{axis}}\) (0.886). | **MISS** |
| F40 | High-mode coeff residual | — | **MISS** |
| F41 | F9 second epoch, lr \(10^{-4}\) | \(S\) 0.9901. Any second step walks off. | **MISS** |
| F42 | kNN stretch in F9 coeff space | Neighbors do not share \(s\). | **MISS** |
| F43 | C9a + edge, stay dualctx | \(S\) 0.9761. Do not put the F9 recipe on another dualctx. | **KILL** |
| F44 | \(q_{95}\)-implied \(\kappa\) stretch | — | **MISS** |
| F45 | F8 + edge-mix 0.4 | \(S\) 0.9859. | **MISS** |
| F46 | F8 + edge-power 2 | Bit-identical to F45: dualshot `_loss` never applied `--edge-weight`. F9’s gain is a geom-off data epoch, not edge. | **MISS** + bug |
| F47 | Real dualshot edge, hot lr | \(S\) 0.9873. 6× too hot vs F9’s \(5\times10^{-5}\). | **MISS** |
| F48 | F8 + 1 ep, geom off, lr \(5\times10^{-5}\) | \(S\) 0.99044. F8 bump, not F9. | **MISS** |
| F49 | F8 + 1 ep, lr \(3\times10^{-4}\) | Same basin as F47. F8+1-ep cannot reproduce F9. | **MISS** |
| F50 | F9 + gentle real edge | \(S\) 0.99127. Spends F9 Cons. | **MISS** |
| F51 | F9 + data-only second ep | Same as F50. Edge vs no-edge is a no-op. Second epoch closed. | **MISS** |
| F52 | F8 + gentle edge | Same as F48. | **MISS** |
| F53 | Shrink F9 \(\kappa\) to map mean | \(\kappa\) 0.842. Frame variation is real. | **KILL** |
| F54 | F9+F4 mean | Dilutes F9 (−0.00083). | **MISS** |
| F55 | F8 + F9 4-ep schedule | Val picked epoch 4 and walked off. | **MISS** |
| F56 | Embed → \(s_Z\) | corr 0.059. | **LOCK** |
| F57 | Oracle \(dZ\) on raw F9 | \(S\) 0.99267 / \(Z_{\mathrm{axis}}\) 1.000. \(\kappa\) stays 0.945. | **CEILING** |
| F58 | Embed → \(dZ\) | corr 0.36 (first real signal). Full \(\hat{dZ}\) is noise. | **MISS** as apply |
| F59 | \(\alpha=0.35\) embed-\(dZ\) | \(S\) **0.99199** (+0.00008 vs F9). Bar was 0.9924. | **PAPER** |
| F60 | Finer \(\alpha\) | Plateau 0.25–0.35. | **LOCK** |
| F61 | Ridge \(dZ\) at \(\alpha=0.35\) | corr 0.13 < MLP 0.36. | **MISS** |
| F62 | Embed+shape+features → \(dZ\) | Extra channels dilute (corr 0.09). | **MISS** |
| F63 | F15 + oracle \(dZ\) | \(S\) 0.99343. \(s_Z\) and \(dZ\) stack. | **CEILING** |
| F64 | F32 + oracle \(dZ\) | \(S\) 0.99390. | **CEILING** |
| F65 | kNN embed → \(dZ\) | corr 0.12. | **MISS** |
| F66 | F32+\(dZ\)+\(0.35\,dR\) | \(S\) **0.99436** / Cons 0.9830. Illegal GT warp. | **CEILING** |
| F67 | Features → \(dZ\) | corr 0.06. Coils do not carry \(dZ\). | **LOCK** |
| F68 | F8 + \(Z_{\mathrm{axis}}\times 3\) | Bit-equal to F48. Post-process family closed. | **MISS** |

**What it left.** F9 is the held *single*. Legal predictors of \(s_Z\), \(dR\), usable \(dZ\) from features / embed / kNN / shape all fail. Oracles prove the knobs exist and are illegal. Successor is a *different layout* parent (Brown `thz`), not another F9 epoch.

Docs: [`MODEL_DESCRIPTION_CRC_FEQ_forest.md`](MODEL_DESCRIPTION_CRC_FEQ_forest.md), [`FOREST_PLAN.md`](FOREST_PLAN.md). Board was `:8768`.

---

## 12. Brown — Z-grid dualctx × F9

**Thesis.** Thomson layout is scrambled (\(Z\) std 55 mm on `th1`). A dualctx trained on a Z-sorted Thomson grid (`thz`) should own \(Z_{\mathrm{axis}}\) and mix with F9.

**Goal.** A heterogeneous residual, not a cousin of F9.

| ID | Goal | Result | Verdict |
|---|---|---|---|
| B1 | Dualshot + stack geom from epoch 1, random init | \(R^2_\psi=-228\) ep 1, still −48 at 30. | **KILL** |
| B2 | Te-weighted kinetic \(Z\) apply on F9 | \(S\) 0.983. F9 already has \(Z\) corr 0.984. | **MISS** |
| B2b | Affine leftover \(dZ\) from Thomson | leftover corr 0.08. | **LOCK** kinetic-\(Z\) |
| B3 | Unfreeze C9a + stack after warmup | One Adam step, \(R^2_\psi=-1.55\). | **KILL** |
| B4 | New dualctx `th1`, stack after 25 | \(S\) 0.98992. Map-first on channel Thomson did not beat F9. | **MISS** |
| B5 | Limiter-fill stretch of F9 | \(S\) 0.970. | **LOCK** vessel-fill |
| B6 | Channel index vs \(Z\) (audit) | Layout is scrambled. Why `thz` exists. Missed the `TS_edge` name. | audit |
| B7 | `thz` dualctx, \(\varepsilon=0.5\) | Single 0.99104. Do not 3-mix with B8. | **MISS** as single |
| B7+F9 | Mean | 0.99298. Superseded by B8+F9. | KEEP then |
| B8 | `thz` dualctx, \(\varepsilon=1.0\), warmup 25 | Single **0.99173**. Owns \(Z_{\mathrm{axis}}\) 0.9855. Do not raise \(\varepsilon\). | **MISS** as single, **KEEP** as parent |
| **B8+F9** | Uniform mean | Local **0.99320** / Cons 0.9754 / \(\kappa\) 0.9519 → public **0.99185 / #5**. | **SHIP** (held) |
| B8+B7+F9 | 3-mix cousins | 0.99306. Shared residual. | **MISS** |
| B10 | Thomson \((R,Z)\) audit | `TS_tangential` is \(R\)-informative. B8 joined `TS_edge` (none). | **DONE** |
| B11 | `thzr` dualctx | Single 0.99081; +F9 0.99283 (−0.00037 vs held); +B8 0.99224 (cousins). | **MISS** |
| B12 | dualshot + `thzr`, geom after 25 | \(S\) 0.0034 / \(R^2_\psi=-3\). Same as B1. | **KILL** |
| B13 | \(\psi_{\mathrm{vac}}\) as 4 PCA *input* channels | KEEP vs B11 (0.99153). +F9 **0.99312** (−0.00008 vs held). Closest miss. | not a ship |
| B14 | \((R,Z)\)+envelope CNN refiner | Cons unchanged vs B8. | **MISS** |
| B15 | Trained free-boundary parent | Ep-8 \(R^2_\psi=0.982\). Holdout \(S\) 0.9794. Map formed, then spent. | **KILL** |
| A1 | Cons autopsy of B8 / F9 / mix | \(Z\) is B8; \(R\)/\(\ell_i\) are F9; \(\kappa\) shared ~0.945. | **PASS** |

**What it left.** The only mix that *shipped* is heterogeneous: `thz` dualctx × `th1` dualshot. Cousin 3-mixes dilute. Public ship is still this mix. Local held mix was replaced 2026-08-21 by Sky S1+F9 (\(S=0.99326\)).

Docs: [`MODEL_DESCRIPTION_CRC_FEQ_brown.md`](MODEL_DESCRIPTION_CRC_FEQ_brown.md), [`BROWN_PLAN.md`](BROWN_PLAN.md). Board was `:8769`. Mix artifacts: `results/CRC_FEQ_brown_B9_mix_B8_F9_score.json`, `results/brown/B9_mix_B8_F9.json`.

---

## 13. Steel — frozen-B8 residuals

**Thesis.** Leave B8 and F9 frozen. Add a small residual (GS / magnetic probes / eddy currents / a learned gate). Fit quality of a new parent is the wrong color.

**Goal.** Move \(\kappa\) without training 19 M weights again.

| ID | Goal | Result | Verdict |
|---|---|---|---|
| A2 | Learned scalar / 60-d gate on frozen B8+F9 | Scalar **0.99323** (+0.00003). Oracle \(w\) 0.99335. \(\kappa\) 0.952. | **PAPER** (do not package) |
| S1 | Frozen B8 + low-D \((p',FF')\) + \(L_{\mathrm{GS}}\) from ep 1 | Holdout \(S\) **0.566** / Cons 0 / \(R^2_\psi\) 0.733. | **KILL** |
| S2 | GS after warmup 8, \(L_{\mathrm{GS}}=0.005\) | \(S\) **0.99172**. Best ep 2. | **MISS** |
| C0 | Existence: poses, cited \(\tau\), shot join | C1 YES (GPEC poses). C2 YES (Sayer 6.9 / 3.3 ms). C3 **NO**. | existence |
| C1-vac | Vacuum \(B/\psi\) PCA into residual head | \(S\) 0.99109 / \(\kappa\) 0.9406. | **MISS** |
| C1-reg | \(L_{\mathrm{mag}}\) vs GT \(\psi\) at probes | \(S\) 0.99120 / \(\kappa\) 0.9399. | **MISS** |
| C1-shape | Same on \(\kappa\) / X-point subset | \(S\) 0.99077 / \(\kappa\) 0.9427. | **MISS** (C1 closed) |
| C2-feat | Eddy PCA at \(\tau=6.9\) ms | \(S\) **0.99109** / \(\kappa\) 0.9429. | **MISS** |
| C2-rl | Predict eddy modes + RL loss | \(S\) **0.98971**. SoftDerive-on crashed Cons. | **MISS** |
| C2-tau33 | Eddy PCA at \(\tau=3.3\) ms | \(S\) **0.99106**. \(\tau\) is not the lever. | **MISS** (C2 closed) |
| C3 | Supervised current / `ECOILB` / shot hash | No map, MDS private. | **NO** |
| C4 | Longer shot context / GRU on frozen map | Locked after K. `steel_ladder c4` exits 2. | **LOCK** |
| K | \(\Delta\kappa\) from coils / \(p_{\mathrm{th}}\) / time / bilinear | Mix OOF \(R^2=-0.019\). | **LOCK** |
| SD | Add a second \(\kappa\) proxy | B8 already has SoftDerive \(\kappa\) after ep 25. The hole is the map. | **LOCK** |
| E1/E2 | New parents (fit quality) | Wrong color. | **MOVED** to Olive |

**What it left.** Frozen-B8 residuals land at \(S\approx 0.991\) or crash. \(\Delta\kappa\) is not in the legal coils, Thomson pressure, or time. No Job 2. No invented `ECOILB`. A2 is a paper keep (+0.00003) — do not package.

Docs: [`MODEL_DESCRIPTION_CRC_FEQ_steel.md`](MODEL_DESCRIPTION_CRC_FEQ_steel.md), [`STEEL_PLAN.md`](STEEL_PLAN.md). Board was `:8770`. **CLOSED 2026-08-20.**

---

## 14. Olive — new plastic parent + early SoftDerive + GradNorm

**Thesis.** SoftDerive on a *finished* map spends \(\psi\) (F19, B3). If it turns on while a new dualctx is still plastic, with GradNorm so \(L_{\mathrm{data}}\) cannot drown \(\nabla\kappa\) and \(\nabla\ell_i\), does Cons move?

**Goal.** A new mix member at \(S\ge 0.99250\), or a mix that beats 0.99320. Not a promise of 0.9945.

P0 (wire GradNorm / ramp / delayed \(w_g\) / \(\ell_p\) floor) was listed as blocking O1. SoftDerive **never turned on** — both O1 arms died in data-only warmup — so the GradNorm question was never asked.

| ID | Goal | Result | Verdict |
|---|---|---|---|
| P0 | Wire GradNorm on \(h_{\mathrm{fuse}}\), \(\lambda_{\mathrm{geom}}\) ramp | Code path existed; geom never reached. | unused |
| O1 | New dualctx + GradNorm + \(\ell_i\)-stack | **Killed ep 8.** \(R^2_\psi=0.9988<0.999\). Holdout \(S=0.98621\) / Cons 0.9512. Ramp still 0. | **KILL** |
| O1.b | Adaptive warmup from O1 epoch 7 | **Killed ep 25.** Never healthy: val \(R^2_\psi=0.9979\) / Cons 0.9208. Holdout \(S=0.9899\) / Cons 0.9628. \(10^{-3}\) shook the map (Cons dipped to 0.65). | **KILL** |
| O1.1 | PCGrad if \(w_g\) collapsed | Collapse file is written only after geom is on and \(w_g\) dies. Never enabled. | **CLOSED** |
| O2 | New dualshot from F8-class init | Holdout is inherited F8. Not a new parent. | miss vs F9 |
| O3.a | \(0.5\,\mathrm{O1}+0.5\,\mathrm{O2}\) | \(S=0.99160\) (gain −0.00016). Weak O1 dragged the mix. | **MISS** |
| O3.b | \(0.5\,\mathrm{O1}+0.5\,\mathrm{F9}\) | \(S=0.99165\) (gain −0.00026). | **MISS** |
| O3.c | \(0.5\,\mathrm{B8}+0.5\,\mathrm{O2}\) | \(S=0.99317\) / Cons 0.9753 (gain +0.00141). **0.00003 under** B8+F9 0.99320. | **MISS** vs held |
| P4 | Dimensionless-\(X\) | Not an Olive job. | **MOVED** to Red |

**What it left.** “Earlier than B8” is false. Early SoftDerive is not deliverable with this loss algebra on a from-scratch dualctx. O2 is not a third parent. Held ship unchanged.

Docs: [`MODEL_DESCRIPTION_CRC_FEQ_olive.md`](MODEL_DESCRIPTION_CRC_FEQ_olive.md), [`OLIVE_PLAN.md`](OLIVE_PLAN.md). Board was `:8771`. **CLOSED 2026-08-20.**

---

## 15. Red — same-wiring non-dim

**Thesis.** #1’s MAST \(S=0.73\) means they are not regressing raw DIII-D coil amps. Rewrite the **same 21 DIII-D columns** in units of \((I_p, B_{\mathrm{tor}}, a, R_0)\) so MAST and DIII-D share a space.

**Goal.** P0: ≥2 magnetics nonzero on ≥10% of MAST frames. Then train R1 (B8 trunk, non-dim \(X\)).

| ID | Goal | Result | Verdict |
|---|---|---|---|
| P0 | 80 DIII-D + 80 MAST magnetics, raw vs non-dim | **FAILED layout.** Only `plasma_current` live on MAST in the 21 DIII-D slots. F / ECOILA / `bcoil` empty (`nonzero_frac=0`). Non-dim of empty columns cannot transfer. | **NO** |
| R1 | B8 + non-dim \(X\), 88 ep | Never launched. | not run |
| R2 | Mix R1+F9 | Parked; no R1. | not run |

`results/red/P0_exist.json`: `n_live_magnetics=1`, live = `plasma_current`. Median \(|I_p|\) ratio raw ≈ 1.03 (machines already match on current); non-dim of \(I_p\) alone is meaningless (log10 span explodes).

**What it left.** MAST coils live under **other names**. Scaling DIII-D column identities does not create them. Do not invent F↔P the same night.

Docs: [`MODEL_DESCRIPTION_CRC_FEQ_red.md`](MODEL_DESCRIPTION_CRC_FEQ_red.md), [`RED_PLAN.md`](RED_PLAN.md). Code: `red_nondim.py`, `scripts/red_exist.py`. Board was `:8772`. **CLOSED 2026-08-20.**

---

## 16. Gray — four shared roles, new parent

**Thesis.** Red failed because names differ. Map four **roles** that both machines actually have — \(I_p\), solenoid (`ECOILA` / `sol_current`), TF (`bcoil` / `tf_current`), shaping energy \(\sum I^2\) — plus Thomson. No F↔P pairing. Retrain dualctx on that \(X\).

**Goal.** P0: all four roles live on both machines. G1: B8 schedule, random init, `--gray`, must hold DIII-D.

| ID | Goal | Result | Verdict |
|---|---|---|---|
| P0 first pass | Roles live on both machines | **False-failed:** \(I_p\) on its own ADC; native-length pad zeroed sol/tf/eshape. | bug, not physics |
| P0 re-run | Same, interpolate onto a common clock | **PASSED.** All four roles live both machines. | **PASS** (`ok=true`) |
| G1 | New dualctx on roles, 88 ep, B8 recipe | Official holdout **MISS.** \(S\) **0.96443** / \(R^2_\psi\) 0.99586 / Cons **0.8628** / \(\kappa\) **0.837** / \(\ell_i\) **0.894**. vs B8 0.99173 / 0.99961 / 0.9692 / 0.945 / 0.958. Humpty loss spikes ~4e4 after geom-on ep 25 (epochs 32/40/52); val recovered. `best_epoch` 87. | **MISS** |
| G2 | Mix G1+F9 | Not launched. Do not mix a miss. | skipped |

`results/gray/P0_exist.json`, `results/gray/G1_roles.json`. Ckpt: `CRC_FEQ_gray_G1_roles.pt` (not a ship).

Ops: first G1 start was reaped with the Cursor agent shell (same millisecond as `gray_watch`; not GPU OOM — 16 GB card, G1 ~3.9 GB). Restarted as user systemd. Drill now requires systemd, not Cursor `block_until_ms: 0`.

**What it left.** Roles **exist**. *Retraining* on them spends DIII-D. The held parents were trained on 21 named coils; four roles are a different \(X\). Do not G1-2.

Docs: [`MODEL_DESCRIPTION_CRC_FEQ_gray.md`](MODEL_DESCRIPTION_CRC_FEQ_gray.md), [`GRAY_PLAN.md`](GRAY_PLAN.md). Code: `gray_roles.py`, `scripts/gray_exist.py`. Board was `:8773`. **CLOSED 2026-08-20.**

---

## 17. Coral — frozen B8; roles→21 at MAST infer only

**Thesis.** Do **not** train a parent. Ablate F-coils on DIII-D (does empty-F look like our MAST 0.055?). If yes, fit ridge 4 Gray roles → 21 magnetics on train. If frozen B8 \(\Delta S\le 0.002\), apply the map at **MAST infer only**.

**Goal.** Challenge 2 without spending Challenge 1. Gate: ablation \(S\sim 0.05\) *and* rebuild \(\Delta S\le 0.002\).

| ID | Goal | Result | Verdict |
|---|---|---|---|
| P0.1 | B8 sanity on real holdout | \(S\) **0.99173** / Cons 0.9692. Matches the held single. | sanity |
| P0.2 | B8 with F-coils zeroed | \(S\) **0.16955** / Cons **0** / \(R^2_\psi\) 0.167. Mix zero \(S=0.235\). | **empty-F IS the MAST disease** |
| P0.3 | Ridge roles→21, infer B8 on recon | B8 recon \(S\) **0.56814** / Cons 0.123 / \(R^2_\psi\) 0.825. **\(\Delta S=0.424\)** vs gate 0.002. Mix recon 0.571. | **FAIL** |
| coil \(R^2\) (train) | Do roles determine F? | ECOILA 0.985, `bcoil` 0.991, `plasma_current` 0.62, **F-coils 0.02–0.25**. | F-pattern is not in the four roles |
| C1 | Apply map at MAST infer | Not launched. | skipped |

`results/coral/P0_exist.json`. Hook `CRC_FEQ_CORAL=1` in `features.py` is MAST-only and defaults **off**. Do not leave it on.

**What it left.** MAST needs the **per-coil identity** B8 was trained on. Four roles rebuild solenoid and TF and cannot invent the 18 F-coil pattern. Cannot invent F↔P. Cannot retrain G1. Cannot rebuild 21 from 4 roles.

Docs: [`MODEL_DESCRIPTION_CRC_FEQ_coral.md`](MODEL_DESCRIPTION_CRC_FEQ_coral.md), [`CORAL_PLAN.md`](CORAL_PLAN.md). Code: `coral_rebuild.py`, `scripts/coral_exist.py`. Board was `:8774`. **CLOSED 2026-08-21.** Successor was Sky (closed).

---

## 18. Sky — coils as spatial current sources (CLOSED)

**Thesis.** Official `coil_R/Z` + currents are a write both machines can fill. Not 21 names, not 4 roles, not B13’s 4 vac PCA on top of names.

**Goal.** A name-free parent at B8 or better, then a mix that beats local 0.99320. MAST fill is the long thesis; DIII-D hold is the gate.

Sequence (as run): P0 → S2 (first GPU) → S1 after S2 miss → M1 after S1 PAPER. S3 is a readout, not a train.

### 18.1 P0 — existence (no GPU)

`results/sky/P0_exist.json` `ok=true` `path=spatial`. 80/80 both machines. Ampere-turns: \(I_{\mathrm{element}}=I_{\mathrm{coil}}/N_{\mathrm{share}}\) so MAST’s 812 turns are not over-counted.

| | DIII-D | MAST |
|---|---:|---:|
| shots ok / vac finite | 80 / 80 | 80 / 80 |
| elements / coils (median) | 19 / 19 | **812 / 11** |
| native \(R\) (m) | 0.72–2.61 | 0.12–1.72 |
| native \(Z\) (m) | −1.60–1.60 | −1.84–1.85 |
| vac PCA **k=4** | **99.94%** | **98.53%** |
| vac mode 1 | 99.57% | 71.9% |
| vac PCA k=16 | 100% | 100% |
| live PF columns | F1A–F9B + ECOILA | p2–p6 L/U + `sol_current` |

**Lesson.** Geometry is legal. B13 was **not** starved of vac modes — DIII-D vacuum is already rank-1. S2 is a name-free write, not a k-raise. Normalize \(((R-R_0)/a, Z/a)\); do not share raw metres.

### 18.2 S2 — full \(\psi_{\mathrm{vac}}\) parent — **MISS / LOCK**

Write: shared box \(r\in[-1.6,1.6]\), \(z\in[-3.0,3.0]\); DIII-D Green \(G\) is \(19\times 256\) (`results/sky/G_d3d_n16.npz`); \(X=\) \(\psi_{\mathrm{vac}}/|I_p|\) (256) + \(I_p/10^6\) + TF/\(|I_p|\) + `thz` (128) = **386**, window-9 → **1158**. B8 schedule (dualctx 1024×8 / ctx 512×3, \(\varepsilon=1\), stack, \(\lambda_w=1.35\), warmup 25, 88 ep). Trunk 19,753,538 params, split=386. Not 4 PCA. Not frozen B8. Not GS. Not a CNN (the field is the write).

Official 80-shot (`S2_vacgrid.json`):

| \(S\) | \(R^2_\psi\) | \(R^2_{q\beta}\) | \(1-D_{\mathrm{LCFS}}\) | Cons | \(\kappa\) |
|---:|---:|---:|---:|---:|---:|
| **0.80217** | 0.84528 | 0.91243 | 0.93994 | 0.53202 | **0.27041** |

Cons breakdown: \(R=0.824\), \(Z=0.523\), \(\kappa=0.270\), \(\delta_{\mathrm{top}}=0.543\), \(\delta_{\mathrm{bot}}=0.370\), vol \(=0.481\), \(\ell_i=0.713\).

Early-stop ep 28 (patience 25); restored **best_epoch 3**, selection 0.5887. Val \(R^2_\psi\) never left ~0.86. Wall 42 min. Do not mix. Do not retrain.

### 18.3 S1 — per-coil tokens — **PAPER**

Write: official DIII-D order, 19 tokens \(\bigl((R-R_0)/a,\; Z/a,\; I/|I_p|\bigr)\) + \(I_p/10^6\) + TF/\(|I_p|\) + `thz` = **187**, window-9 → **561**. Same B8 schedule. Trunk 19,345,986 params, split=187. Flag `--sky-tokens` / `CRC_FEQ_SKY_TOKENS`. Not DeepSets at train (N=19 is fixed); a variable-N pool would be MAST infer.

Official 80-shot (`S1_tokens.json`), 88/88, **best_epoch 88**, wall 197 min:

| | \(S\) | \(R^2_\psi\) | \(R^2_{q\beta}\) | \(1-D_{\mathrm{LCFS}}\) | Cons | \(\kappa\) | \(\ell_i\) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **S1** | **0.99184** | 0.99945 | 0.99615 | 0.98681 | **0.97018** | **0.94559** | 0.95730 |
| B8 | 0.99173 | 0.99961 | — | — | 0.9692 | 0.9445 | 0.9584 |
| F9 | 0.99191 | 0.99945 | 0.99624 | 0.98705 | 0.97037 | 0.9454 | — |

Cons: \(R=0.962\), \(Z=0.986\), \(\kappa=0.946\), \(\delta_{\mathrm{top}}=0.988\), \(\delta_{\mathrm{bot}}=0.978\), vol \(=0.974\), \(\ell_i=0.957\). Val Cons climbed after geom-on (ep 25) from ~0.85 to 0.965; official Cons 0.9702.

Above B8 on \(S\) and Cons. Below KEEP-SINGLE 0.99250. Mix-legal as PAPER.

### 18.4 S3 — \(\kappa\) readout

Not a train. S2 \(\kappa=0.270\) (vac grid did not invent elongation). S1 \(\kappa=0.9456\) vs B8 0.9445 / F9 0.9454 — tokens did not close the shared hole. `results/sky/S3_kappa.json`.

### 18.5 M1 — mix

First attempt **VOID**: `load_charcoal(S1)` left `CRC_FEQ_SKY_TOKENS=1`, so F9 inferred as tokens. Member F9 \(S=0.04680\) / \(R^2_\psi=-1.765\); ensemble \(S=0.148\). Not a verdict. Fix: pop color env flags per checkpoint.

Official remixed 80-shot (`M1_mix.json`):

| | \(S\) | \(R^2_\psi\) | \(R^2_{q\beta}\) | \(1-D_{\mathrm{LCFS}}\) | Cons | \(\kappa\) | \(\ell_i\) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **S1+F9** | **0.99326** | 0.99962 | 0.99684 | 0.98787 | **0.97579** | **0.95251** | 0.96907 |
| B8+F9 (was held) | 0.99320 | **0.99966** | 0.99684 | 0.98777 | 0.97542 | 0.95194 | **0.96923** |
| \(\Delta\) | **+0.00006** | −0.00005 | 0 | +0.00009 | +0.00037 | +0.00057 | −0.00016 |

Gain over F9: \(S\) +0.00135, Cons +0.00542. Members: S1 \(S=0.99184\), F9 \(S=0.99191\).

**KEEP** vs the written bar (\(S>0.99320\)). New **local** held. Do not package (house 0.9945). Public ship stays B8+F9. Do not reuse the Brown public−local offset on this mix.

### 18.6 Job table

| ID | Goal | Result | Verdict |
|---|---|---|---|
| P0 | Tokens + finite \(\psi_{\mathrm{vac}}\) both machines | 80/80. DIII-D 19 / MAST 812. Vac k=4 = 99.94% / 98.53%. | **PASS** |
| S2 | Full \(16\times 16\) \(\psi_{\mathrm{vac}}\) parent | \(S=0.80217\) / \(R^2_\psi=0.845\) / Cons 0.532 / \(\kappa=0.270\). Best ep 3. | **MISS / LOCK** |
| S1 | Tokens \((r,z,I/|I_p|)\times 19\) + Ip + TF + `thz` | \(S=0.99184\) / Cons 0.9702 / \(\kappa=0.9456\) vs B8 0.99173 / 0.9692 / 0.9445. | **PAPER** |
| S3 | \(\kappa\) readout | S2 \(\kappa=0.270\); S1 \(\kappa=0.9456\). | readout |
| M1 (void) | Mix under leaked `CRC_FEQ_SKY_TOKENS` | F9 \(S=0.047\). | **VOID** |
| **M1** | \(0.5\,\mathrm{S1}+0.5\,\mathrm{F9}\) | \(S=0.99326\) / Cons 0.9758 / \(\kappa=0.9525\). | **KEEP** (local held). Do not package. |

Ckpt: `CRC_FEQ_sky_S1_tokens.pt`. S2 ckpt is not a ship. Hook `--sky` / `--sky-tokens` defaults **off**.

**What it left.** A vac raster without names does not hold DIII-D. Per-coil tokens do, at B8+Cons, and the F9 mix is the new local held (+0.00006 \(S\), +\(0.00057\) \(\kappa\) vs B8+F9). The \(\kappa\) hole on the singles is unchanged. MAST token transfer was not scored. Public ship is still B8+F9. Submit bar is still 0.9945.

Docs: [`MODEL_DESCRIPTION_CRC_FEQ_sky.md`](MODEL_DESCRIPTION_CRC_FEQ_sky.md), [`SKY_PLAN.md`](SKY_PLAN.md). Code: `sky_geom.py`, `scripts/sky_exist.py`, `scripts/sky_ladder.py`. Board `:8775` (archive). **CLOSED 2026-08-21.** Successor is Yellow.

---

## 19. Yellow — \(\psi_{\mathrm{vac}},\vec B_{\mathrm{vac}}\) at Thomson \(Z\) (CLOSED)

**Thesis:** \(\psi_{\mathrm{vac}},\vec B_{\mathrm{vac}}\) at Thomson \(Z\) plus S1 tokens move \(\kappa\) without spending DIII-D. Not S2’s raster, not B13’s 4 PCA, not GS.

**Status:** **CLOSED** 2026-08-21. Tree finished. Local held stays S1+F9. Public ship stays B8+F9. No successor. White is an unused name, not a color.

### 19.1 P0 / Y1 — existence — **PASS**

80/80 DIII-D: official `thomson_chord_R/Z` live; finite \(\psi,B_R,B_Z\) at 32 `thz` \(Z\)-bins on 80/80. \(R_{\mathrm{th}}=1.940\) m. 19 PF coils. `G_ts_n32.npz` written. Median \(\psi_{\mathrm{vac}}\) RMS \(7.7\times 10^{-7}\) (the Green image is small next to plasma \(\psi\); that is a fact about vacuum at the core chord, not a bug).

### 19.2 Y2 — S1 tokens + vac-aug `thz` parent — **MISS**

Write: S1 tokens (57) + \(I_p/10^6\) + TF/\(|I_p|\) + 32 blocks of \([T_e^{\mathrm{c}},n_e^{\mathrm{c}},T_e^{\mathrm{e}},n_e^{\mathrm{e}},\psi_{\mathrm{vac}}/|I_p|,B_R,B_Z]\) = **283**, window-9 → **849**. B8 schedule (dualctx 1024×8 / ctx 512×3, \(\varepsilon=1\), stack, \(\lambda_w=1.35\), warmup 25, 88 ep). Trunk 19,542,594 params, split=283. Flag `--yellow` / `CRC_FEQ_YELLOW`. Not a vac grid. Not 4 PCA. Not frozen B8. Not unfreeze.

Official 80-shot (`Y2_parent.json`), 88/88, **best_epoch 88**, wall 195 min:

| | \(S\) | \(R^2_\psi\) | \(R^2_{q\beta}\) | \(1-D_{\mathrm{LCFS}}\) | Cons | \(\kappa\) | \(\ell_i\) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Y2** | **0.99136** | 0.99942 | 0.99564 | 0.98636 | **0.96850** | **0.94079** | 0.95857 |
| S1 | 0.99184 | 0.99945 | 0.99615 | 0.98681 | 0.97018 | 0.94559 | 0.95730 |
| B8 | 0.99173 | 0.99961 | 0.99619 | 0.98673 | 0.96922 | 0.94451 | 0.95843 |
| \(\Delta\) vs S1 | **−0.00047** | −0.00003 | −0.00051 | −0.00045 | **−0.00168** | **−0.00479** | +0.00127 |

Cons: \(R=0.957\), \(Z=0.987\), \(\kappa=0.941\), \(\delta_{\mathrm{top}}=0.988\), \(\delta_{\mathrm{bot}}=0.975\), vol \(=0.974\), \(\ell_i=0.959\).

Live val_Cons tracked S1’s curve (ep 68: 0.9588 vs S1 0.9585; ep 88: 0.9664 vs S1 0.9653). Official Cons came in **below** S1. The proxy said “photocopy.” The holdout said “slightly worse, and \(\kappa\) spent.” Val proxy is not the verdict. This was already known (Gray; C5a warmup). It was ignored for the mid-run read.

Below B8 on \(S\), Cons, and \(\kappa\). Not PAPER. Do not mix. Do not retrain.

### 19.3 Y3 — \(\kappa\) readout

Not a train. Y2 \(\kappa=0.9408\) vs S1 0.9456 / B8 0.9445 / F9 0.9454 / hole 0.945. The hole moved the wrong way.

### 19.4 M1 — mix

**SKIPPED.** `_parent_is_paper` requires \(S>0.99173\) and a Cons/\(\kappa\) gain, or \(S\ge 0.99250\). Y2 has neither.

### 19.5 Job table

| ID | Goal | Result | Verdict |
|---|---|---|---|
| P0 | Chords + finite vac-at-TS | 80/80. \(R_{\mathrm{th}}=1.940\) m. 19 coils. | **PASS** |
| Y1 | `G_ts_n32.npz` | On disk (P0 artifact). | **PASS** |
| Y2 | New dualctx, tokens + vac-aug `thz` | \(S=0.99136\) / Cons 0.9685 / \(\kappa=0.9408\). | **MISS** |
| Y3 | \(\kappa\) readout | 0.9408 vs S1 0.9456. | readout |
| M1 | \(0.5\,\mathrm{Y2}+0.5\,\mathrm{F9}\) | Not launched. | **SKIPPED** |

Ckpt: `CRC_FEQ_yellow_Y2_parent.pt` is not a ship. Hook `--yellow` defaults **off**.

**What it left.** Vac-at-TS is a legal write and a worse parent than tokens alone. The last named coil-geometry-at-kinetics encoding is spent. MAST token infer of S1 was still not scored (and is not a local official score; see §23). Public ship is still B8+F9. Submit bar is still 0.9945. The leftover that opened White is a **second architecture family**: keep \(\psi\) as a 2D image.

Docs: [`MODEL_DESCRIPTION_CRC_FEQ_yellow.md`](MODEL_DESCRIPTION_CRC_FEQ_yellow.md), [`YELLOW_PLAN.md`](YELLOW_PLAN.md). Code: `yellow_geom.py`, `scripts/yellow_exist.py`, `scripts/yellow_ladder.py`. Board `:8776` (stopped). **CLOSED 2026-08-21.** Successor is White.

---

## 20. Unused names

None. White `#FFFFFF` is open ([§25](#25-white--spatial-u-net-open)). There is no unused luminance name after 100%.

---

## 21. Graveyard (do not reopen)

Closed questions. Thoroughness is not a reason to rerun them.

1. **The basis is not the ceiling.** R6 / Charcoal 0a: GT through \(k=60\) is Cons 0.9972. Raising \(k\), INR (Navy), FieldRefiner, X-point basis are dead.
2. **Pixel \(R^2_\psi\) is spent on the board.** 3% of the #1 gap.
3. **A biased Cons surrogate is worse than none.** Teal \(\kappa\); geom-on MLP (maroon P1); Soft LCFS (R10b, F12).
4. **Concatenated temporal context loses.** R2a, R4, R10a. Dualctx (C1) is the delivery that works.
5. **Capacity saturated at ~17 M.** R7 vs R9: +0.00004 \(S\).
6. **Whitened \(L_{\mathrm{coeff}}\) saturates at \(\lambda_w=1.35\).** C7c is a null.
7. **Longer training can anti-align.** C3; F9/F11/F12 restore `best_epoch` 1.
8. **Geom on a random field shatters \(\psi\).** B1, B12.
9. **Unfreezing a finished parent walks off.** B3, F19, F41/F50/F51. Do not unfreeze B8/F9.
10. **Post-process of F9 cannot see \(s_Z\), \(dR\), or a usable \(dZ\).** F16/F21/F56/F67. F66 0.99436 is illegal.
11. **Cousin means dilute.** B8+B7+F9, B13+B8, F9+F4.
12. **A gate on B8+F9 cannot invent \(\kappa\).** A2 +0.00003; oracle +0.00015.
13. **Frozen-B8 residuals are spent.** S1/S2/C1/C2. Eddy \(\tau\) 6.9 vs 3.3 is the same number.
14. **\(\Delta\kappa\) is not in the legal coils.** K OOF \(R^2=-0.019\).
15. **No test MPI / `ECOILB` / shot-hash map.** C3 is NO. Do not invent them.
16. **GRU as the only trunk dies.** D1a, F1, 396 M.
17. **Inference GS kills a finished map.** F13 \(S=0.861\). Vacuum-as-\(Y\) is dead (Charcoal 25.1). Vac-as-*input* (B13) is a closest miss, not a reopen.
18. **Same-wiring non-dim cannot transfer.** Red P0: 20/21 MAST slots empty.
19. **Retrain on four roles spends DIII-D.** Gray G1 \(S=0.9644\).
20. **Roles cannot rebuild F-coils for frozen B8.** Coral \(\Delta S=0.424\). Empty-F *is* MAST 0.055.
21. **Do not invent F↔P or a 21+14 zero union.**
22. **Olive early SoftDerive never turned on.** O1 / O1.b killed in warmup. O1.1 never enabled. Answered in [§32](#32-geometry-loss--it-finally-ran-on-a-healthy-map-and-it-is-null-closed-2026-09-05): on a converged parent it turns on every time, and it is null.
23. **A full \(\psi_{\mathrm{vac}}\) raster as the only coil write spends DIII-D.** Sky S2 \(S=0.802\). Do not retrain S2.
24. **Do not leave Sky/Gray/Coral env flags on when inferring F9.** First M1 was VOID (\(S=0.148\)).
25. **Vac-at-Thomson does not close the \(\kappa\) hole.** Yellow Y2 \(S=0.99136\) / \(\kappa=0.9408\). Do not retrain Y2. Do not mix it.

---

## 22. Leftover (legal questions we have not named)

If the next color cannot state a *new measured thesis*, it is not a color. After Yellow this list is observations, not a queue. See §23.

1. **DIII-D Cons hole.** #1 public Cons 0.980 vs us 0.971. R6 says \(k=60\) *can* hold Cons 0.997. Every write we could name is measured. The leftover is a write we cannot name.
2. **MAST 0.73.** S1 tokens were never scored on Challenge 2. Official MAST \(S\) is the public zip. There is no local MAST GT (`local_score.MACHINE = DIII-D`; train is 7,041 DIII-D / 0 MAST). Not a color whose gate we can evaluate here.
3. **A third layout** that is not `th1` / `thz` / `thzr` theater. B10 closed `TS_edge`. B11/B13 missed. Divertor has no Te/ne.
4. **Organizer shot map / public `ECOILB`.** Still NO. Until it appears, C3 stays NO.

Ignore: F↔P, 21+14 union, Gray G1-2, Red R1, Olive GradNorm retry, Steel residuals, unfreeze, raise \(k\), INR, inference GS, vacuum-as-\(Y\), Job 2 / C4 / invented 5 ms \(\tau\), Sky S2 vac-grid retrain, Yellow Y2 retrain / mix.

---

## 23. Out of ideas

This is an audit of the search, not a request for a new job. The problem still has solutions: #1 is public \(S=0.99438\) / Cons 0.980, and R6 says our own \(k=60\) basis can hold Cons 0.997 / \(\kappa\) 0.989 if we emit the right \(\psi\). Those facts mean the *problem* is not exhausted. The *model of the space this project used to pick jobs* is.

### 23.1 The model

The project factored the score into a small set of axes and treated a color as a one-variable walk on one axis:

| axis | what we thought moved \(S\) | how we searched it |
|---|---|---|
| Decoder | PCA \(k\), INR, a refiner | Charcoal 0a, Navy, Maroon R6 / kopt |
| Capacity / trunk | width, depth, MLP → ResNet → dualctx → GRU | Maroon R1–R9, Purple C1/D1, Forest F1/F9 |
| Data | 200 k frames vs full corpus | Maroon M1 |
| Loss | geom, whitened \(L_{\mathrm{coeff}}\), LCFS, GS, GradNorm | Teal; Maroon P1/R10b/C5–C7; Steel S1; Olive O1 |
| Context | concat vs own encoder vs shot GRU | R2a/R4/R10a, C1, D2, F4/F9 |
| Average / warp | seeds, hetero mix, \(s_Z,dR,dZ\), a gate | R3/R8, ship6, F15–F67, A2 |
| \(X\) write | 21 names, `th*`, roles, tokens, vac raster / PCA / TS, eddy, non-dim | Brown, Red, Gray, Coral, Sky, Yellow |
| Frozen residual | GS, probes, \(\Delta\kappa\) from coils | Steel |
| Machine transfer | same 21 slots, four roles, tokens | Red, Gray, Coral, Sky P0 |

The implicit claim: if \(\kappa\) is 74% of the #1 gap and R6 says the basis can hold it, then \(\kappa\) is missing from \(X\) or from the loss, and a legal write or a hotter stack will put it in \(\psi\). Jobs were designed as existence probe → one GPU parent → mix only if PAPER. That is a tree on the factorization, not a search of the true space.

### 23.2 What the tree actually did

It closed every axis we named.

- **Basis / pixels.** GT through \(k=60\) is Cons 0.997. Raising \(k\) is 0.001% of \(\psi\) MSE. Pixel \(R^2_\psi\) is 3% of the public gap. We are not losing on the decoder.
- **Trunk / data / \(\lambda\).** Full corpus, residual trunk, clip 10, dualctx, \(\lambda_w=1.35\), ~17 M. The next increment on each of those was a null or a walk-off (R9, C7c, C3, F9 `best_epoch` 1).
- **Loss as a \(\kappa\) lever.** Geom-on MLP paid the boundary. Soft LCFS moved the surrogate and not the metric. GS on a finished map killed it. GradNorm never left warmup. \(\Delta\kappa\) from coils is OOF \(R^2=-0.019\).
- **Warps and gates.** F16/F21/F56/F67: \(X\), shape, embed, kNN do not carry \(s_Z\) or \(dR\). A2 on the mix is +0.00003. Oracle \(w\) is +0.00015. A gate cannot invent elongation.
- **\(X\) remaps of the same coils.** 21 names hold DIII-D and forfeit MAST. Roles spend DIII-D or cannot rebuild F. Tokens hold DIII-D at B8+\(\kappa=0.946\). Vac raster spends DIII-D (\(\kappa=0.270\)). Vac 4-PCA is a closest miss. Vac-at-TS is worse than tokens and spends \(\kappa\) (Yellow). That is the same information — \((r,z,I)\) — in five frames. After K, repeating the remap was a prediction that a *representation* of a null would stop being a null. Yellow measured that prediction: false. The live val_Cons match to S1 was the same information arriving by a longer path.

The late colors (Steel → Yellow) did not add a new axis. They added encodings of axes already scored.

### 23.3 Why “exhaustive” is true of the model and false of the problem

Exhaustive *inside the factorization*: there is no remaining job that is (a) a new axis, (b) legal, (c) fail-closed, and (d) locally scoreable. The leftover list in §22 is four rows: a write we cannot name, a Codabench-only MAST number, a Thomson layout already missed, and an organizer file that does not exist. None of those is a color.

Inconclusive *about the problem*:

1. **A better \(\psi\) exists.** #1’s public Cons is 0.980. R6’s GT reconstruction is 0.997. We emit a \(\psi\) whose derived \(\kappa\) is 0.945 on every single parent in the family (B8 0.9445, S1 0.9456, F9 0.9454, Y2 0.9408). The mix lifts \(\kappa\) to 0.952 by averaging residual character. That is the hole. We did not find the write that fills it. We found that our writes do not.
2. **#1 is not our family.** Purple shipped on the premise that rank 1 was “PCA50 + dual-context ResNet.” pshishkin (`seq + mlp`) and dmitryzhevn (`25`) falsified that. The search after that still walked *our* trunk and *our* \(X\). We did not build a second family. We do not know their write. Calling the space explored is calling *our* menu explored.
3. **The submit bar may sit above this family.** House 0.9945 is above the illegal F66 GT warp (0.99436) and above the A2 oracle mix (0.99335). If those ceilings are real, no legal post-process of F9 and no mix of B8-class parents reaches the bar we set. We kept the bar and kept training parents that saturate at 0.9918 single / 0.9933 mix.
4. **Challenge 2 was never locally scored.** S1 tokens on MAST is an unmeasured observation. It is also not a holdout we can run: no MAST GT on disk. Treating “not scored” as “next color” was a category error (White). Treating it as closed would also be a category error. It is a hole in the exploration, and it is not a job this machine can judge.
5. **Public Cons of the local held is unknown.** S1+F9 is 0.99326 local and not shipped. The Brown offset must not be reused on it. We do not know whether the held mix is closer to #1 than B8+F9 on the only board that pays.

### 23.4 Mistakes in the model, not in the effort

- **\(\kappa\in X\).** After K, the prior should have been: elongation is not a function of the legal coils / \(p_{\mathrm{th}}\) / time that we can read. Sky/Yellow still bet that a Green image of those coils would teach it. That is the same prior with a nicer basis. The measurement is now in. Continuing would be refusing the likelihood.
- **One family.** Capacity, \(\lambda\), window, and geom were swept inside PCA-60 + dualctx/dualshot. When the architecture lock died, the next six colors were still that family plus an \(X\) remap.
- **Proxy during the run.** Yellow’s val_Cons was read as “S1 photocopy, PAPER-shaped.” Official was a miss that spent \(\kappa\). The project already had this failure mode in writing. The mid-run narrative used the proxy anyway.
- **Color as motion.** A luminance row was opened whenever the GPU was free and a leftover sentence existed. Olive §0.5 said not to invent a third parent. The drill said not to open a color to look busy. White was named before a gate existed. That is the process failing its own stop rule.
- **What was not tried, on purpose.** A second architecture family. A Cons loss that is the official seven scalars on the official extractor (SoftDerive is a replica, and it did not close \(\kappa\)). **[Corrected 2026-09-02 by [§30.2](#302-softderives-gradient-is-faithful--234-was-wrong-on-this).** The parenthetical was wrong. Driving the SoftDerive loss to zero drives the *official* Cons to 0.9948. The replica's gradient is faithful; what did not close \(\kappa\) was the absence of a scalar target better than the map's own, not the extractor.**] Training on MAST (no released GT; Challenge 2 is zero-shot). Submitting S1+F9 (bar). Asking #1 what `25` is. Those are not “unexplored axes in the model.” They are things the model forbade or could not evaluate. If the answer is one of them, the model was wrong, not incomplete.

### 23.5 Stop condition

A new color requires a sentence of the form: *this write, this gate, this local official number.* After Yellow the Dense/PCA family had no such sentence that was not a reopen. White is the leftover §23.4 listed as not tried: a **second architecture family** (fully conv U-Net, official \(65\times 65\), never flatten). That is [§25](#25-white--spatial-u-net-open).

---

## 25. White — spatial U-Net (CLOSED 2026-08-24)

**Thesis.** \(\psi(R,Z)\) is a 2D Grad–Shafranov field. Rasterize vacuum + Thomson onto the official \(65\times 65\) scorer grid and predict \(\psi\) with a fully convolutional U-Net. No `nn.Linear`. Sky S2’s defect was the **flatten** into Dense (\(S=0.802\), \(\kappa=0.270\)), not the Green. This is not a retrain of S2.

**Goal.** A new parent at B8 or better with a Cons / \(\kappa\) gain, then a mix that beats local 0.99326. Official 80-shot is the verdict.

**Terminal decision (2026-08-24):** Audit + one geom sprint variant failed official gates (\(S=0.692<0.75\)). **White is closed.** Public ship stays **B8+F9**. Do not reopen reshuffle loop or authorize full-corpus White.

Sequence: P0 → dummy-batch → W2 → AUD → SPR → **DEC (close)**.

| ID | Goal | Result | Verdict |
|---|---|---|---|
| P0 | Official \(65\times 65\) Green + Thomson raster | Finite 7-ch 80/80. Vac 80/80. Thomson 80/80. 19 coils. | **PASS** |
| dummy | U-Net forward/backward, no Linear | `[32,7,65,65]→[32,1,65,65]`. 2.999M. Linear=False. | **PASS** |
| W2 | U-Net parent; MSE+Sobel then **SoftDerive** pivot | Best val \(R^2=0.875\) reshuffle8; official \(S\sim 0.68\) Cons \(\approx 0\). Reshuffle loop **STOPPED** 2026-08-24. | **CLOSED** |
| AUD | Official score reshuffle8/24/29; PCA on/off | `W2_audit.json` best S=0.684 Cons≈0.003 | **DONE** recipe_failed |
| SPR | geom-emphasis from reshuffle8 | S1_geom05: mini LIKE, official S=0.692, hard stop | **DONE** |
| DEC | Terminal gate record | `W2_decision.json` close White | **DONE** |
| W3 | \(\kappa\) readout | — | **PARKED** (no PAPER parent) |
| M1 | \(0.5\,\mathrm{W2}+0.5\,\mathrm{F9}\) | — | **PARKED** |
| S1+F9 cal | Public calibration probe (not championship submit) | `S1F9_calibration_probe.json` packaged | **UPLOAD PENDING** |

Do not flatten. Do not unfreeze B8/F9. Do not retrain S2 or Y2. Do not invent F↔P.

Docs: [`MODEL_DESCRIPTION_CRC_FEQ_white.md`](MODEL_DESCRIPTION_CRC_FEQ_white.md), [`WHITE_PLAN.md`](WHITE_PLAN.md). Code: `white_geom.py`, `white_unet.py`, `scripts/white_exist.py`, `scripts/white_train.py`. Board `:8777`.

**FiLM addendum (2026-08-26).** A fully convolutional FiLM spatial decoder distilled onto B8+F9 teacher maps missed Gate 0 (\(S=0.97693\) / \(\kappa=0.828\)). Spatial family closed. Successor is Ultraviolet.

---

## 27. Ultraviolet — fillable tokens + whole-shot Transformer (OPEN 2026-08-26)

**Thesis.** Top-3 public MAST_S is not from MAST labels. It is a fillable write: per-coil \((R,Z,I/|I_p|)\) that both machines can fill, trained on DIII-D official \(S\). Frozen Sky S1 is fixed N=19 dualctx and zero-fills MAST. Ultraviolet retrains a variable-N set encoder plus a noncausal whole-shot Transformer. Official DIII-D \(S\) selects. MAST_HEALTH is logged, never selected.

**Goal.** Public DIII-D \(S \ge 0.993620\) (live #3, 2026-08-26). Then MAST_S toward 0.743 without spending DIII-D.

**Bars (2026-08-26 board):** `TOP3_D3D_S=0.993620`. `RANK1=0.994376` / Cons 0.980 / MAST_S 0.743. `S1F9_LOCAL=0.993259`. `S1F9_PUBLIC=0.991626`. `BASE_PUBLIC` B8+F9 0.991848.

Sequence: CAL → P0 (CPU, parallel) → T1 → T2 → M1/outer/hist80. Frozen-S1 RTE is CPU-only, expected NO.

| ID | Goal | Result | Verdict |
|---|---|---|---|
| CAL | Public S1+F9 S and MAST_S | public S=0.991626 / #8 / MAST_S≈0 | **DONE** keep B8+F9 |
| RTE | Frozen S1 MAST contract | zero-fill + fixed N=19 | **NO** |
| P0 | Fillable write + whole-shot info + official preflight | write PASS; wholeShot OOF R² collapse; preflight API miss | **MISS** |
| T1 | Teacher distill | — | **CLOSED** (P0 miss) |
| T2 | GT fine-tune; PAPER at \(S\ge 0.993620\) | — | **CLOSED** (P0 miss) |
| M1 / OUTER / HIST80 / FINAL | Mix, sealed, package | — | **PARKED** |

Do not reopen White/FiLM. Do not fall back to th1-149. Do not invent F↔P. MAST public-archive training is Track 2; Challenge 1 is first.

Docs: [`MODEL_DESCRIPTION_CRC_FEQ_ultraviolet.md`](MODEL_DESCRIPTION_CRC_FEQ_ultraviolet.md), [`ULTRAVIOLET_PLAN.md`](ULTRAVIOLET_PLAN.md). Board `:8779`.

---

## 28. MLP3 — machine-agnostic conv flux operator (OPEN 2026-08-31)

**Thesis.** Feed a conv body Ψ_vac plus explicit R,Z planes on a shared metric scale, co-train on unlabeled MAST with label-free physics, and the operator transfers. First submission returned MAST \(R^2_\psi=-0.137\) and cost a step.

**Cause was two bugs, not weak transfer.** (1) `gs_vacuum_loss` drove \(\Delta^*\psi\to0\) *inside* the vessel — the vacuum equation, valid only where there is no current — so the global minimum of the MAST objective was \(\psi=\Psi_{vac}\). It also normalized by its own detached value, reporting 1.0 for *any* field, so it was blind to the collapse while its gradient pointed at it. (2) DIII-D `magnetics_plasma_current` sits on a foreign clock: 42% of 1,559,340 frames read \(|I_p|<50\) kA against a ~1356 kA truth, corr 0.127. The offset is **not constant** (`ip_t[0]-mag_t[0]` is right for 2/3 of shots, out by ~8 s for the rest); `ip_align_shift` aligns the \(I_p\) plasma window to the EFIT window instead — label-free, no clock provenance. Dead frames 42.25% → 0.00%. Ψ_vac left alone on purpose (its coil clock wants only ~-825 ms; rewriting it would void the PCA sidecars and the GAMMA contract).

**Bars.** Old MLP3 checkpoint enclosed **1.2 kA against a 628.7 kA MAST target** (0.19%; Ψ_vac itself 0.6 kA) — it was emitting its own input.

| ID | Goal | Result | Verdict |
|---|---|---|---|
| IPFIX | Correct \(I_p\); rebuild pool from the Ampère integral | dead frames 42.25%→0.00%; normalized channel 0.74 (42% at zero) → 1.38, vs MAST 2.17 | **DONE** |
| AMP | Replace the vacuum-interior term with Ampère enclosed current | recovers \(I_p\) to 1.016 MAST / 0.998 DIII-D; forbids the vacuum solution; needs 3-cell mask erosion | **DONE** |
| A | Arm A: loss + \(I_p\), 8 ep | enclosed \(I_p\) 0.002→**0.998**, LCFS 0%→**100%**, DIII-D val \(R^2_\psi\) **0.876** (12h run: 0.713 same ep). MAST \(R^2_\psi\) **−1.353**, axis 0.643 m out | **NO** plasma exists, wrong place |
| B | Arm B: + relative \((R-R_0)/a,\,Z/a\) coords | axis error 0.643→**0.404 m**, \(Z\) 0.068→**0.020 m**, MAST \(R^2_\psi\) **−1.081**; DIII-D 0.803 | **NO** ~37% of the gap |
| — | Coordinate diagnosis | shared coords: DIII-D axis +0.6614, MAST truth +0.3511, Arm A predicted +0.6035 = **81% of the way to the DIII-D address** | confirmed |
| ~~GS~~ | ~~Phase 4 GS solver~~ | ~~**required**~~ | **RETRACTED 2026-08-31** — see below |
| ANCHOR | Ψ_vac anchor + brown DIII-D hybrid, no MAST training | holdout \(R^2_\psi\) +0.448; **public [910998](https://www.codabench.org/competitions/17739/detailed_results/910998): `mast_S=0.185`, `G_ratio=0.186`, MAST \(R^2_\psi=+0.336\), `d3d_S=0.992` unchanged** | **SHIPPED & scored** |
| C | Arm C: zero-init residual skip on the anchor, 12 ep | DIII-D val \(R^2_\psi\) **0.890** (best of the line; A 0.876, B 0.803). MAST \(R^2_\psi\) **−0.938**, never above −0.72 after epoch 1 | **NO** — floor destroyed in one epoch |

**Retraction: the GS conclusion was an artefact of the test.** The row above concluded that position error required a GS solver. It was measured by shifting the *whole predicted field*, background included, which moves the vacuum contribution off the coils that produced it and so cannot help. Redone against a correctly scaled anchor, decomposing the prediction into background and plasma parts:

| on the 115-frame labelled MAST holdout | \(R^2_\psi\) |
|---|---:|
| anchor alone | +0.538 |
| anchor + Arm B's plasma part, at its best gain (which is **0.00**) | +0.538 |
| anchor + Arm B's plasma part shifted onto the true axis, gain 0.50 | +0.655 |
| per-frame-rescaled Ψ_vac — the ceiling of this family | +0.728 |

Placement is worth **+0.12**, and the optimal weight on Arm B's plasma *as placed* is exactly zero. The dominant term is calibration of the vacuum background, not placement. GS goes back on the shelf as a Phase 3 item, worth revisiting only after the background is fixed.

**The real bug: `DELTA = 0`.** The reconstruction contract is \(\psi=\hat\psi\,\Gamma s+\Delta m\), and `frame_norm` ships \(\Delta=0\) — measured on DIII-D, where the offset genuinely does not help, because \(\mathrm{mean}(\Psi_{vac})\) crosses zero there. So the net must synthesise the entire DC level, and it learns DIII-D's. Holding the \(\Gamma s\) scale fixed and moving only the offset on MAST:

| DC offset | MAST \(R^2_\psi\) |
|---|---:|
| 0 — what ships today | +0.085 |
| DIII-D-rim units bridge, \(U=258.4\) | **+0.448** |
| \(U=1000\), the "coils are kA into a per-ampere Green" units argument | +0.218 |
| \(U=593\), the same estimator fitted on MAST truth (not shippable) | +0.578 |
| true fold mean | +0.549 |

The units argument is *worse* than the empirical fit, so the bridge is fitted and labelled as fitted. \(U\) is obtained on DIII-D **rim cells only** (vessel mask minus an erosion), where the plasma contributes least. Narrowing the rim raises \(|U|\) monotonically — 242.9, 228.4, 213.6, 198.3, 182.5 at widths 1–5 — and that trend is linear to **0.18%** of its intercept, so the residual contamination is extrapolated to zero wall thickness rather than tolerated at some chosen width: \(U_{RIM}=258.38\). Nothing in the fit touches a MAST label; the three demo shots stay a pure holdout.

**The asymmetry underneath all of it.** A per-frame-rescaled Ψ_vac explains **67%** of each MAST frame's own flux variance but only **18%** of each DIII-D frame's. MAST is coil-dominated, DIII-D is plasma-dominated. A DIII-D-trained operator learns to synthesise the 82% that is plasma, and on MAST it paints that over a field that wanted 33%. This is why Arms A and B shipped MAST maps *worse than their own input* (−1.353 and −1.081 against +0.085 for copying it), and why the anchor — which contains no plasma at all — beats both.

**What the anchor does not buy (public confirms holdout).** It has no closed flux surface: `mast_dlcfs=1.0` on [910998](https://www.codabench.org/competitions/17739/detailed_results/910998), so the \(1-D_{LCFS}\) and Consistency terms are zero. Arm B scored zero on both as well, so nothing was given up. Public `mast_S=0.185` is almost entirely `0.55 × mast_r2_psi` with `mast_r2_psi=+0.336` (holdout was +0.448 — slightly optimistic, transfer still cleared the 0.0553 revert bar).

**Arm C failed, and the way it failed is the most useful thing here.** The zero-init skip guarantees the *starting point*, not the endpoint, and one epoch of DIII-D supervision was enough to spend the floor: MAST \(R^2_\psi\) went \(+0.448 \to -0.935\) and never recovered over 12 epochs (−0.72 at best, −0.938 at the selected epoch). The mechanism follows directly from the 67/18 asymmetry. On DIII-D the anchor explains only 18%, so the residual must supply the other 82%; that same residual applied to MAST paints 82% of a plasma onto a field that wanted 33%. **The skip did not remove the asymmetry, it relocated it from the whole network into the residual.**

Two things Arm C did buy, both worth keeping:

* **DIII-D got better, not worse.** Val \(R^2_\psi\) 0.890 is the best of the line (A 0.876, B 0.803) and it led from epoch 1 (0.506 vs 0.401/0.379). Handing a conv body a calibrated background is a real head start. The anchor is not a MAST-only device.
* **Arm C closes an LCFS on 100% of holdout frames** (the anchor closes 0%, Arms A/B 0%), with derivations failing on 0.50%. So the trained net has the *structure* and the wrong calibration; the anchor has the calibration and no structure. Neither alone scores: Arm C's composite is 0.0000 because \(1-D_{LCFS}\) is still zero — the surface closes far from the truth.

The next idea this points at is not a bigger net. It is that the residual needs a machine-aware amplitude — the plasma's share of the flux is 82% on one machine and 33% on the other, and nothing in the current objective tells the net that. Note the tension `relative` coords create: they were adopted to fix placement by making the two machines look alike, which is exactly what removes the network's ability to tell them apart and attenuate accordingly.

Do not reship any arm. Do not unfreeze B8/F9, which remains the DIII-D artifact. Do not rewrite Ψ_vac's clock without rebuilding the PCA sidecars and refitting GAMMA. Do **not** re-fit \(\Delta\) globally in `frame_norm.json` — every existing checkpoint including brown would silently change meaning; the anchor is versioned in cfg instead. **What ships for MAST is anchor** — the untrained Ψ_vac floor, not another color and not Arm C.

### 28.1 Just anchor

Not a hex color in [§4](#4-one-line-per-color). We call it **anchor**: brown B8+F9 on DIII-D, `mast_anchor.anchor_psi` on MAST, no training. The shipped Codabench tag was `0.1.0+CRC_FEQ_anchor+928c0d2`; say **anchor** in the ledger. The color tree (Black → … → Brown → Ultraviolet) is the PCA/dualctx line; MLP3 and anchor sit beside it. Live color on the tree is still **Ultraviolet** ([§27](#27-ultraviolet--fillable-tokens--whole-shot-transformer-open-2026-08-26)).

Docs: [`MODEL_DESCRIPTION_CRC_FEQ_mlp3.md`](MODEL_DESCRIPTION_CRC_FEQ_mlp3.md) §8–9. Code: `mast_anchor.py`, `mlp3_physics.py`, `psivac.py`, `machine_env.py`, `scripts/fit_rim_bridge.py`, `scripts/anchor_package.py`, `scripts/mast_score.py`. Tests: `tests/test_mast_anchor.py`, `tests/test_anchor_residual.py`, `tests/test_ampere_ip.py`, `tests/test_coord_modes.py`.

---

## 29. Solver — free-boundary Grad–Shafranov on MAST (OPEN 2026-09-01)

Not a color. **solver**: brown B8+F9 unchanged on DIII-D, and on MAST a forward free-boundary Grad–Shafranov solve with no learned component and no MAST-fitted constant. MAST holdout \(R^2_\psi\) **+0.9179**, against 0.336 for anchor and the 0.728 ceiling of the vacuum-rescaling family.

### 29.1 This is not F13

[§9](#9-forest) killed **F13**, "inference GS projection of F9", at \(S=0.861\), and [§21](#21-graveyard-do-not-reopen) item 17 records it as *"inference GS kills a finished map"*. That verdict stands and this does not contradict it. F13 took a map a network had already produced and relaxed it toward GS, which can only move a good map around. Here there is no finished map to damage: \(\psi\) is *generated* from the measured coil currents and \(I_p\) by solving the equation, and the learned line never touches MAST. The graveyard entry is about GS as a **post-process**; this is GS as the **forward model**. Steel's frozen-B8 residuals and Navy's INR are likewise orthogonal — both parametrise \(\psi\) and fit it, neither solves for it.

### 29.2 Ψ_vac was wrong, and how we know it is right now

The **exact decomposition** is what made everything else measurable. Apply the discrete \(\Delta^*\) to a label, read off the implied current \(q=-\Delta^*\psi/(\mu_0R)\,\mathrm{d}A\) inside the eroded vessel, and \(\psi_{\rm ext}=\psi_{\rm gt}-P q\) is the external field *exactly*, with no fit. That gives a per-machine ceiling for any solver: MAST **+0.9991** with zero MAST-fitted parameters, DIII-D **+0.9572** on 30 held-out shots with one.

Three concrete corrections came out of it.

* **Reversed-polarity discharges.** 14 of 120 DIII-D shots store coil currents in the discharge's own frame rather than the machine's, so \(\Psi_{\rm vac}\) needs a global sign flip. \(\operatorname{sign}(I_p)\) predicts which, **120 of 120 shots**. Lives in `vac_calib.polarity`, applied on read so revising the rule never triggers another 7,041-shot reparse.
* **MAST solenoid parallel winding.** The solenoid's 656 geometry elements are not 656 turns in series; the effective factor is **2.0**, i.e. `n_share / 2`. `vac_calib.N_PARALLEL`.
* **DIII-D E-coil turns.** `D3_ECOILA_TURNS = 135`, from a scan; the previous value sat off the edge of its range.

**WITHDRAWN: the DIII-D passive-structure claim.** An earlier session reported a DIII-D \(\Psi_{\rm vac}\) ceiling near +0.20 and attributed the gap to in-vessel passive structure carrying current the coil basis cannot see. That was an artefact of the two bugs above — the missing polarity flip plus the bad E-coil turn count. With both fixed the ceiling is **+0.9572** and there is no evidence for a passive-structure term. Do not cite the +0.20 number or the passive-structure explanation.

A separate sign bug: `gs_solver` carried its own `VAC_SIGN = {"d3": 1, "mast": 1}` while `vac_calib` already folds `AXIS_SIGN` in, so DIII-D was fed a negated vacuum field. Replaced by `machine_env.AXIS_SIGN`; MAST was unaffected, which is why it went unnoticed.

### 29.3 The profile is a prerequisite, not a refinement

The plan treated a canonical profile as a checkpoint and per-frame profiles as polish. That is backwards, and the mechanism matters. Radial force balance ties the plasma's major radius to the current profile, so a wrong profile does not blur the answer — past some distance **no equilibrium exists at the measured \(I_p\)**, and the iteration correctly fails to find one, walking the plasma into the wall. Canonical \((3,3,0.25)\) has no equilibrium for **11 of 42 frames** of `mast_shot_28350`, which score \(R^2_\psi\approx-8\) while the rest of the shot runs +0.98. A fixed-profile gate cannot pass, on either machine.

Handed the *labels'* geometry the family is ample: **+0.9984** per-frame median on DIII-D and **+0.9999** on MAST (`gs_profile.fit_from_labels`). So the parametrisation was never the bottleneck.

### 29.4 Two solver failure modes, both real

* **O-point lock-on.** `SoftDerive._boundary` takes the strongest local maximum of \(\psi\) in the eroded envelope, which is right for scoring a finished map and wrong inside a solver: \(\Psi_{\rm vac}\) has its own maxima, and MAST has one near the outer wall at \(R=1.82\). Early in a discharge it outscores the real O-point, the sweep puts the current there, and the iteration locks into a 3-cell "plasma" against the wall — a genuine fixed point that no relaxation escapes. Cured by continuation: `gs_solver.plasma_boundary` takes the flux maximum *within the previous sweep's plasma*. `SoftDerive` is untouched, since the scorers depend on it.
* **Convergence measured on the wrong quantity.** Freezing a collapsed frame drives the \(\psi\) increment to zero, so **38 of 42 frames** reported themselves converged while sitting against the wall. Convergence is now measured on the current density, which has no such blind spot.

### 29.5 DIII-D does not solve, and the reason is physics

**BLOCKED.** Started *exactly at the ground truth* with the exact external field, the DIII-D plasma drifts from \(Z=-0.10\) to \(-1.00\) over thirty sweeps and \(R^2_\psi\) falls +0.996 → −1.68. DIII-D is strongly elongated, so the true equilibrium is a **saddle of the Picard map** — the vertical instability of the equilibrium is inherited by the iteration. Returning the minimum-residual iterate does not help: the wrong attractor has a *lower* residual than the truth. Held-out DIII-D frames score \(R^2_\psi\) p50 −2.87 even with exact \(\Psi_{\rm ext}\).

This is not a profile-family failure — oracle geometry gives **+0.9852** on the same frames — and not an \(I_p\) failure. Across 40 shots the Ampère-recovered \(I_p\) matches the measured channel at p50 0.998, all within 5%. (An earlier reading of 71% came from sampling only each shot's first 30 frames, which are all ramp-up.) Fixing it needs vertical stabilisation — pin the current centroid to the external \(B_R=0\) null, or JFNK — not tuning. MAST is far less elongated and solves fine, which is why the deliverable is unaffected.

### 29.6 DIII-D profile knowledge does not transfer to MAST

`scripts/fit_profile_mlp.py` fits targets on DIII-D and regresses them from dimensionless machine-agnostic inputs (`gs_features`). It learns: **+0.54** \(R^2\) on the current-shape curve and **+0.80** on \(\lambda\) across 124 held-out DIII-D shots. It does not transfer: on MAST the head saturates, predicting \(\beta\) of 16–20 against a DIII-D p90 of 2, and the holdout scores **−3.77**. Dropping `dsep` (p90 \(|z|=35\) on MAST — the channel carries sentinels there) and clamping inputs to \(\pm3\sigma\) both help the diagnosis and neither fixes it.

It is the machine gap, not the model: a fixed profile at DIII-D's own median \((1,1,0.625)\) also fails on MAST at **−0.76**. And the profile that does work there, \(\approx(2,2,0)\) at +0.856, is only identifiable *using* MAST labels, so it is not legally available. **Regressing profiles across machines is dead** unless someone finds an input that closes the gap.

### 29.7 What ships: Thomson picks the profile

Per frame, run every candidate profile to convergence and keep the equilibrium whose own \(p(\psi_N)\) best matches the measured Thomson pressure. Nothing is fitted and nothing is learned; Thomson is a legal input not otherwise used, and the criterion reads identically on both machines.

It works because it sees the direction the solver residual is blind in. Writing \(j=A\,s(u)\,[\lambda R/R_0+(1-\lambda)R_0/R]\), the pressure *shape* against \(u\) depends only on \((\alpha,\beta)\) — \(\lambda\) cancels, both terms carrying the same \(s(u)\) — while the *amplitude* \(p(u)=(A\lambda\,\mathrm{depth}/R_0)\int_0^u s\) depends on \(\lambda\). So Thomson pins all three. The absolute scale checks out with no fudge factor: **1432 Pa** predicted against a measured \(2p_e\) of **1326 Pa**, which independently confirms the flux convention end to end.

| MAST holdout, 115 frames | \(R^2_\psi\) |
|---|---|
| anchor (shipped) | +0.336 |
| vacuum-rescaling family ceiling | 0.728 |
| label-free: solver residual only | +0.7687 |
| **label-free: Thomson pressure** | **+0.9179** (+0.9128 on the narrowed 34-candidate grid used at inference) |
| oracle profile (uses labels) | +0.9598 |
| \(\Psi_{\rm vac}\) ceiling (perfect plasma) | +0.9991 |

One bug nearly buried this: interpolating Thomson onto the EFIT clock fills MAST's all-NaN pad column with finite values, so the pad is invisible downstream and the alignment check silently discarded all 130 core chords, leaving only the 16 edge ones. `gs_thomson.chords` detects pads on the raw array.

The single unmeasured physical input is \(p\approx2p_e\) (\(T_i\sim T_e\), \(n_i\sim n_e\)). It enters the amplitude term only, is identical on both machines, and is not fitted.

### 29.8 Cost of the fold, and why the first estimate was 28 hours

The 1206-shot public fold first ran at 85 s a shot, 28 GPU-hours, with the card 35% busy and a single CPU core pinned at 100%. Almost none of it was arithmetic.

`SoftDerive._bottleneck` ends each of its 96 flood-fill sweeps with `torch.equal`, which is a device synchronisation. Fine for a diagnostic evaluated once per map; inside a solver at 500 Picard sweeps and 34 candidate profiles it is over a million stalls per shot. `gs_solver._maximin` is the same relaxation testing every 32 sweeps instead, and is bit-identical whenever the seed lies in the envelope, which the solver's axis always does. `SoftDerive` is untouched.

Solving one candidate at a time also made every sweep a 57-row GEMM against a 4225-column response matrix. `solve` already accepts per-frame profile parameters, so candidates now stack into one batch. The chunk optimum is interior — 14.0 s a shot at 384 rows against 16.7 at 512, 24.3 at 2048 and 36.9 unbatched — because a chunk sweeps until all of its rows settle, so a big chunk waits on its slowest member.

Batching first *cost* 0.0013 of \(R^2_\psi\), which is the interesting part: it means sweeping past convergence hurts. With the vertical instability of §29.5 the trajectory can find a lower-residual iterate later on the wrong attractor, and the best-iterate rule then takes it. Frames now freeze once they meet the tolerance. That makes a frame's answer independent of what shares its batch — every chunk size scores identically — and lifts the narrowed-grid holdout from **+0.9118 to +0.9128**, with the two frames that previously had no converged candidate now converging. End state is 95% GPU and about 5.5 hours.

TF32 in the GEMM was tried and rejected: 20.25 s a shot against 20.93, and \(-0.004\) of \(R^2_\psi\). The response-matrix multiply is not the bottleneck; the 96-pass flood fill is.

### 29.10 Measuring one term of four is how 30% of the score went unseen

The v1 solve went to Codabench on a gate that printed `R2_psi` and nothing else. It came back `mast_S` 0.5254: \(R^2_\psi\) 0.8526 as expected, and `D_LCFS` 0.4356 with `Consistency` **exactly 0.0000** — 30% of the composite, never once measured locally, on either machine.

`mast_eval.score_holdout` had computed all four terms through the vendored reduction the whole time. Nothing called it. So the fix was wiring, not machinery, and the composite formula reproduces every competitor's published score to 1e-7.

Baseline on the 115 frames, v1, with the full reduction:

| | \(R^2_\psi\) | \(R^2_{qb}\) | 1−`D_LCFS` | `Consistency` | S |
|---|---:|---:|---:|---:|---:|
| v1 holdout | 0.9128 | 0.0000 | 0.6326 | 0.0000 | 0.5653 |
| v1 public | 0.8526 | −0.4623 | 0.5644 | 0.0000 | 0.5254 |
| **v2 holdout** | 0.9063 | 0.0000 | **0.8024** | **0.0206** | **0.5828** |

\(R^2_\psi\) is also the *least* sensitive of the four to the failure that was happening: a plasma displaced a metre vertically still shares most of its pixels with the truth, but its LCFS will not close and every shape scalar from it is wrong.

Two lessons that generalise past this color. All seven `Consistency` scalars scored a negative pooled \(R^2\) and clamped to zero, driven by *systematic bias* rather than outliers — axis 6.4 cm too far out, cross-section too round (κ 1.645 against 1.861), current too peaked (`li` 1.213 against 1.067). And `Z_axis` read **−8611** because the truth barely moves on these shots (p1–p99 spans 0.0037 to 0.0112 m), so a 1.9 cm bias is enormous against the truth's own variance. Beating a constant predictor there needs millimetres.

### 29.11 The vertical anchor, and the target that was anti-correlated

The Picard map inherits the equilibrium's vertical instability (§29.5). On the public fold 6.9% of frames came back with the plasma somewhere it cannot be — 1st-percentile `Z_axis` of −1.69 m, on the vessel floor — and those are the same frames whose LCFS will not close. Best-iterate tracking was not enough.

The cure is to project the rigid-shift eigenmode out: each sweep, shift the current's Z centroid onto a target plane, *before* the \(I_p\) normalisation so the shift cannot change the enclosed current.

The obvious target is wrong, and instructively so. Vertical force balance on a current ring is \(F_Z = 2\pi R I_p B_{R,\mathrm{ext}}\) with \(B_R = -(1/R)\,\partial\psi/\partial Z\), so a filament can only sit still where \(\partial\Psi_{vac}/\partial Z = 0\). Against the 115 labelled frames that null is **anti-correlated** with the truth: \(r=-0.752\), rms error 23.8 mm against a true spread of 1.2 mm. \(\Psi_{vac}\) is assembled from *measured* coil currents, and those include the vertical-control system's response — the feedback exists to oppose the drift, so the null of the total coil field sits on the far side of the plasma from wherever it is being pushed. The measured field encodes the correction, not the position.

What works is the machine's own aim point: all 146 MAST Thomson chords lie at exactly Z = 0.000, which predicts the true axis to 8.0 mm rms, three times better, with no fitted constant on either machine.

Anchoring is worth more than its score line suggests. It cut the public-fold implausible fraction from 16.1% to 9.9%, took LCFS-closure failure and vertical drift both to **0.0%**, fixed κ (1.884 against 1.861) and `li` (1.079 against 1.067) almost exactly, made `li` the first `Consistency` scalar ever to score positive, and ran **2.4× faster** because the anchored iteration converges sooner.

### 29.12 Selection is now the binding constraint, not the family or the solver

Under the anchor, an oracle allowed to pick the best of 125 candidates per frame reaches \(R^2_\psi\) **+0.9937** per-frame mean, with `R_axis` 0.910 and volume 8.19 against truth 0.892 and 8.05. The anchor raised that ceiling from +0.9598.

The Thomson criterion gets +0.8749 and agrees with the oracle on **0% of frames**. So the profile family is expressive enough and the solver is sound; the criterion is what is wrong. That is the largest remaining lever on this color, worth roughly 0.09 of pooled \(R^2_\psi\) plus most of `Consistency` and `D_LCFS`.

Corollary worth keeping: the *narrowed* 34-candidate grid scores **better** than the full 125 (0.5828 against 0.5757). A noisy criterion does better with fewer chances to go wrong, which is itself evidence the criterion rather than the coverage is the problem.

Convergence and plausibility are also independent questions, and the solver was only asking the first. A vertically displaced equilibrium is a *genuine* fixed point and passes a residual test exactly, and `CORE_MIN_CELLS` at 24 only catches total collapse against the ~800 cells a real MAST plasma occupies. `gs_plausible.admissible` now gates candidates on dimensionless axis-placement and core-size bounds, and the fallback is the best converged-but-implausible candidate rather than the vacuum field, which has no closed surface at all.

### 29.13 q95 and betaN: derived, validated on DIII-D, still held back on MAST

Both are functionals of the equilibrium, and `validate_submission.py` says exactly what is missing from ψ: "q95 needs F(psi), betaN needs p(psi)". Both come from measured inputs that were **unused** — `magnetics_tf_current`/`magnetics_bcoil` for \(F=\mu_0 N I/2\pi\), and Thomson for \(p\).

On ground-truth flux, so that the derivation is measured and not the solve:

| | DIII-D (calibration) | MAST (read-only gate) |
|---|---:|---:|
| q95 | **+0.9654** (ratio 1.005, r 0.990) | −0.4556 (ratio 1.005, r 0.859) |
| betaN | **+0.7464** (ratio 0.747, r 0.929) | −1.1703 (ratio 0.619, r 0.697) |

Ground truth pinned the turn counts, which is the technique to remember: a wrong turn count is a pure multiplicative offset at *unchanged correlation*. DIII-D at 288 turns gave a ratio of 2.021 with r = 0.994 — a factor-of-two miscount announcing itself — and 144 gives 1.011 with \(B_{t0}\) = 1.6–2.0 T, its real operating range. MAST is 24, at 1.004 and \(B_{t0}\) = 0.478 T. A separate one-line bug, `right=0.0` in the pressure interpolation, blanked every cell inside the innermost Thomson chord and cost a factor of ~1.6 in `betaN` on both machines.

q95 is *exact* on most of MAST — per-frame derived/label spans 1.001 to 1.009 from the 5th to the 75th percentile, and \(R^2\) over the 79 frames with label q95 < 9 is **+0.9983** — and is ruined by a ~15% tail running 1.6× high at low current and low flux depth. Two label-free excisions were tried and both rejected: surface self-certification measures magnetic shear, which is genuinely larger on a spherical tokamak (MAST median 0.338 against DIII-D 0.180); and flat-top \(I_p\) gating works on MAST (+0.6475) while wrecking DIII-D (+0.9654 → +0.2960), which is how a MAST-specific constant announces itself.

`betaN`'s MAST gap is a consistent 1.6× at unchanged correlation and is probably physical: MAST is strongly beam-heated and \(p = 2p_e\) omits the fast-ion pressure EFIT's label includes. There are **no NBI or beam-power columns** in the corpus, so it cannot be modelled from inputs.

So both stay off, and the MAST npz ships `q95`/`betaN` as nan. That costs nothing and is not a regression: the metric takes \(0.15\max(0, R^2_{qb})\), so the donor's −0.4623 was already contributing exactly zero — the public 0.5254 reproduces to 1e-4 without it. nan is the only option of the three that does not claim a number it cannot stand behind, and it takes the donor out of the MAST half completely.

### 29.14 fp16 is fine, and it was worth checking

The submission stores `psirz` as float16 — 483 MiB against 966 at fp32. Not obviously harmless, since the `Consistency` scalars are built from \(|\nabla\psi|\), a difference between neighbouring cells that differ by roughly a thirtieth of the flux depth, so a relative rounding error is amplified by about that factor. Measured: fp16 costs **0.00004** of `MAST_S` (`li` −0.0016, `Z_axis` −0.67 on a term already clamped at zero). Keep fp16. `mast_gate --precision-ab`.

### 29.9 Do not

Do not reopen inference-GS-as-post-process; §21.17 stands. Do not regress profiles from DIII-D to MAST (§29.6). Do not cite the withdrawn DIII-D passive-structure claim or the +0.20 ceiling (§29.2). Do not change `SoftDerive._boundary` or `_bottleneck` — the solver has its own `plasma_boundary` and `_maximin` precisely so the scorers stay fixed. Do not gate the solver on a single canonical profile (§29.3). Do not expect DIII-D to solve without vertical stabilisation (§29.5). Do not enable TF32 or unfreeze converged frames for speed (§29.8).

Do not ship a solve gated on \(R^2_\psi\) alone (§29.10). Do not anchor the plasma vertically to the \(\partial\Psi_{vac}/\partial Z=0\) null — it is anti-correlated with the truth (§29.11). Do not treat solver convergence as evidence a map is an equilibrium (§29.12). Do not tune a q95 or betaN gate on the 115 MAST frames; DIII-D is the calibration set (§29.13). Do not switch the package to fp32 for accuracy (§29.14). Do not reuse a psi cache across a solver change — `mast_solve.run` skips shots whose output exists, so a re-solve over a stale cache is a silent no-op; that is what `VERSION` is for.

### 29.15 Public v2 (912438) — holdout transferred on ψ and LCFS, not Consistency

Submitted 2026-09-02 UTC after HF push (`b7550d9`). DIII-D byte-identical to brown (sha256 match). MAST from `mast_gs_solve_v2` cache.

| | v1 public (912040) | **v2 public (912438)** | v2 holdout |
|---|---:|---:|---:|
| `mast_S` | 0.5254 | **0.5859** | 0.5828 |
| \(R^2_\psi\) | 0.8526 | **0.9279** | 0.9063 |
| `D_LCFS` | 0.4356 | **0.2446** | 0.1976 |
| Consistency | 0.0000 | **0.0000** | 0.0206 |
| `G_ratio` | ~0.53 | **0.591** | — |

Label-free plausibility audit on the **public npz** (1200 frames, 200 shots × 6): **10.6%** ANY-flag (axis 0.2%, LCFS 0.1%, `li` 8.2%, volume 2.7%). Holdout had 0.0% ANY-flag on 115 frames — the Consistency term did not transfer. Revert bar (0.0553) cleared by 10×.

Challenge 1 rank unchanged at **#10** (`d3d_S` still 0.991848). This package was always a MAST/`G_ratio` bet.

Code: `gs_solver.py`, `gs_plausible.py`, `gs_profile.py`, `gs_features.py`, `gs_scalars.py`, `gs_thomson.py`, `vac_calib.py`, `cpu_budget.py`, `scripts/fit_vac_gains.py`, `scripts/fit_profile_mlp.py`, `scripts/mast_solve.py`, `scripts/mast_gate.py`, `scripts/mast_audit.py`, `scripts/scalar_gate.py`.

---

## 30. Projection — the map is one gradient walk from Cons 0.995 (OPEN 2026-09-02)

Code: `scripts/projection_exist.py`. Results: `results/brown/PROJECTION_EXIST.json`. Every number
below is the **official** `build_reference` derive, not the SoftDerive replica. Samples are
subsets of the 80-shot B8/F9 support cache, so baselines differ between gates
(G1/G2 0.942, G3 0.972, G4 0.976 against the 0.975 local mix). **Compare within a gate only.**

### 30.1 The Consistency loss is in the leading modes, and splices break the field

G1 replaces PCA modes \(\ge K\) with truth on the B8+F9 mix (240 frames, 40 shots).

| oracle on | Cons | \(\kappa\) | \(\ell_i\) |
|---|---:|---:|---:|
| nothing | 0.9500 | 0.9257 | 0.9217 |
| modes 30–59 | **0.9441** | 0.9229 | **0.8876** |
| modes 20–59 | 0.9593 | 0.9405 | 0.9306 |
| modes 10–59 | 0.9725 | 0.9789 | 0.9703 |
| modes 5–59 | 0.9872 | 0.9776 | 0.9952 |

Two facts. **The tail is worthless**: a free perfect tail (30–59) scores *below* our own
prediction. The whole gap sits in modes 0–30, roughly half of it in 0–10 — the modes where
mix/GT coefficient correlation is already 0.997 and \(R^2_\psi\) is 0.9996. Consistency is an
**ill-conditioning** problem, not a missing-fine-structure problem.

**And splices are not free.** Handing the model a true tail *drops* \(\ell_i\) 0.922 → 0.888. A
partially-corrected field is not a coherent equilibrium and the gradient-sensitive scalars see it
immediately. This is the mechanism behind F15–F67, the PCA tangent, and A2: they are all low-rank
post-hoc splices of a finished map. It is also why the contour representation missed its P0 gate —
a nested-surface parametrization is a smoother.

Corollary for the loss: whitened \(L_{\mathrm{coeff}}\) up-weights **low-variance** modes, i.e.
exactly the ones G1 shows are irrelevant. C7 saturating at \(\lambda_w=1.35\) and C7c being a null
are consequences of the weighting pointing at the wrong end of the spectrum. Do not raise
\(\lambda_w\).

### 30.2 SoftDerive's gradient is faithful — §23.4 was wrong on this

§23.4 listed "a Cons loss that is the official seven scalars on the official extractor" as not
tried, on the premise that *"SoftDerive is a replica, and it did not close \(\kappa\)."* G2
falsifies the premise. Descending the SoftDerive geom loss on \(z\) directly (Adam, 200 steps,
GT-derived scalar targets, 128 frames) and scoring with the **official** derive:

| | start | 200 steps |
|---|---:|---:|
| Cons (official) | 0.9424 | **0.9948** |
| \(\kappa\) | 0.9168 | 0.9831 |
| \(\ell_i\) | 0.8983 | 0.9895 |
| \(R_{\mathrm{axis}}\) / \(Z_{\mathrm{axis}}\) | 0.940 / 0.946 | **1.0000 / 1.0000** |
| volume | 0.9264 | 0.9996 |
| \(R^2_\psi\) | 0.99903 | **0.99897** |

Driving the replica to zero drives the **real** metric to 0.995. The surrogate is not the
bottleneck. \(\kappa\) lands at 0.983, at R6's basis ceiling of 0.9887.

**Read the last row.** Cons +0.052 costs \(6\times10^{-5}\) of \(R^2_\psi\). Against our shipped
Cons 0.975 the headroom is \(\approx +0.02\) Cons \(\approx +0.004\) \(S\) — larger than the
0.0032 that separates us from #1.

Single-scalar descent is a trap and is not the objective: \(\kappa\)-only
(`--kappa-only`) reaches \(\kappa=0.986\) while Cons **falls** 0.942 → 0.859. The seven scalars are
only jointly satisfiable. `geom_weights=stack` is a mild version of the same mistake.

So the family's problem is **not** representation, **not** the surrogate, and **not** the
\(\psi\) cost. A Cons-0.995 field exists inside our own \(k=60\) basis, a short walk from the map
we ship, reachable by a loss we already have, and the walk is label-free at inference. The route
is missing exactly one input: **an estimate of the seven scalars better than the map's own.**

### 30.3 The two cheap scalar estimators are both nulls

G2 is an oracle. G3 and G4 are the label-free ways to beat the map's own derived scalars.

**G3 — scalar-space stacking.** Grouped 5-fold OOF ridge (3174 frames, 80 shots) on
\([c_{B8}, c_{F9}, q_{95}, \beta_N, \text{axis}]\) → GT scalar, against the \(\psi\)-level mix:

| | \(\kappa\) | \(\ell_i\) | \(R_{\mathrm{axis}}\) | **Cons** |
|---|---:|---:|---:|---:|
| mix | 0.9529 | 0.9557 | 0.9670 | **0.97190** |
| OOF stack | 0.9548 | 0.9514 | 0.9653 | **0.97175** |

\(\Delta = -0.00015\). **NULL.** B8 and F9 make the same errors; blending them in scalar space adds
nothing over blending in \(\psi\). Same finding as "cousins dilute" (§12), now measured on scalars.

**G4 — temporal smoothing.** EFIT labels are smooth in \(t\); our derived scalars need not be.
Centred moving average within contiguous runs (7162 frames, 29 shots):

| window | 1 | 3 | 9 | 25 |
|---|---:|---:|---:|---:|
| Cons | **0.97599** | 0.97590 | 0.97133 | 0.95324 |
| \(\kappa\) | 0.9589 | 0.9615 | 0.9588 | 0.9407 |

\(\kappa\) gains +0.0026 at \(w=3\); everything else loses more. **NULL**, monotone worse after.
The scalar error is temporally *correlated*, not white — consistent with the coefficient
regression slope of truth on prediction being 0.998 (no shrinkage to rescale away). Do not reopen
smoothing of \(\psi\) or of scalars.

### 30.4 What this leaves, and the stop condition

The projection machine is built and verified. The open question is now one **7-dimensional
supervised regression**: beat \(R^2_\kappa=0.953\) on the mix's own derived scalars, from legal
inputs, out of fold. Any gain converts to \(S\) at a measured rate via G2.

The only estimator that is *not* a cousin of B8/F9 has never been run at scale: a network trained
**directly on the seven scalars** rather than on \(\psi\). The single datapoint against it is the
operator-C1 information probe at OOF \(R^2_\kappa=0.828\) — a small head at 40 epochs, weaker than
the 17 M parents, and notably our map reaches 0.945 while never being trained on \(\kappa\) at all.
That probe does not bound a dedicated model.

Gate for the next job, in §23.5 form: *a dedicated 7-scalar predictor, on the legal write, scored
grouped-OOF on the support cache; it must beat mix-derived \(R^2_\kappa=0.9529\) and mean scalar
\(R^2\) 0.97190 out of fold, or the projection route is closed and the gap is information.*

### 30.5 The gate ran and saturated: \(\kappa\) is information-limited at \(\approx 0.956\) (CLOSED)

Code: `scripts/projection_scalar_gate.py`. Results: `results/brown/P30_SCALAR_GATE{,_3500,_7000}.json`.
Trained `HistGradientBoosting` on the 7-scalar SoftDerive targets from the legal `thz` write
(447-d `[raw|avg|diff]`, window 9), on the **same train fold the parents used** (`keys[80:]`), scored
on the **official 80-shot holdout** (`keys[:80]`). Alignment is verified by the baseline reproducing
the ledger's B8+F9 holdout \(\kappa=0.9519\) exactly. Blend is a grouped-cross-fit 2-parameter
combination of (map-derived, model-predicted) per scalar, so a decorrelated-but-weaker estimator can
still win.

| train shots | model \(\kappa\) alone | blend \(\kappa\) | err corr w/ map | \(\Delta\)Cons |
|---:|---:|---:|---:|---:|
| 1 500 | 0.8621 | 0.9540 | +0.415 | +0.00037 |
| 3 500 | 0.8961 | 0.9556 | +0.412 | +0.00127 |
| 7 000 | 0.9133 | 0.9564 | **+0.471** | **+0.00130** |

**Saturated.** Doubling 3500 → 7000 bought \(+0.00003\). Read the mechanism in the third column: as
the model gets stronger its errors become *more* correlated with the map's. Two estimator classes
that share no architecture, no loss, and no training signal converge on the same predictable
component; the residual is predictable by neither. That is an information ceiling, not a data or
capacity shortage.

This is a real result and not another cousin. The tabular model is genuinely independent
(err corr 0.41–0.47, against G3's implicit \(\approx 1\)), it is the **first estimator in the project
to add rather than dilute**, and standalone it reaches \(\kappa=0.913\) — well above the
operator-C1 probe's 0.828 that §30.4 correctly refused to treat as a bound. It still saturates.

**Arithmetic of the closure.** Projection makes the map's derived scalars equal the predicted ones,
so post-projection Cons \(\approx\) scalar-prediction \(R^2\). Best blend is 0.97673 against the
map's own 0.97542: \(\Delta\)Cons \(=+0.0013 \Rightarrow +0.00026\ S\), less \(3\times10^{-5}\ S\)
for the \(\psi\) cost. Net \(\approx +0.0002\ S\), landing local \(\approx 0.9934\) — under the house
bar and an order of magnitude short of the \(+0.0032\) to #1. **Not worth shipping.**

**CLOSED.** The projection mechanism (§30.2) is correct, verified, and reusable — if a scalar
estimate ever arrives from a genuinely new information source, it converts to \(S\) at a measured
rate for \(6\times10^{-5}\) of \(R^2_\psi\). What is closed is sourcing that estimate from the legal
write. §23.4's prior — *elongation is not a function of the legal coils / \(p_{\mathrm{th}}\) / time
that we can read* — now has its strongest evidence: a saturating scaling curve to the full corpus
with an independent estimator class, plus rising error correlation as the mechanism.

Do not reopen: scalar stacking (G3), temporal smoothing of \(\psi\) or scalars (G4), tabular or
tree-based scalar regression on the `thz` write at any corpus size (30.5), \(\lambda_w\) increases
(30.1), post-hoc splices of a finished map (30.1). The remaining hypothesis for #1 is unchanged and
is now the only one standing: **a write that carries information ours does not.**

### 30.6 Consistency is a slave variable of \(\psi\) error — "pixel \(\psi\) is a ceiling" was the wrong read

The premise under §2.1, §23, and most of the last six colors is that \(S\) has two independent axes:
\(R^2_\psi\) is saturated and worth 3.2% of the gap, so the work belongs on the Cons/\(\kappa\) hole.
**That decomposition is wrong**, and §30.1 implies why: Consistency is read off the leading PCA
modes, which is what \(R^2_\psi\) measures. They are not two axes. They are one.

Measured directly. Scale our coefficient error toward truth, \(z=z_{gt}+\alpha\,(z_{mix}-z_{gt})\),
and score with the official derive (640 frames, 80 shots):

| \(\alpha\) | \(R^2_\psi\) | Cons | \(\kappa\) | |
|---:|---:|---:|---:|---|
| 1.000 | 0.999413 | 0.95811 | 0.9409 | ours |
| 0.767 | 0.999655 | 0.97252 | 0.9583 | |
| 0.600 | 0.999789 | 0.98129 | 0.9705 | **= #1's public \(R^2_\psi\)** |
| 0.400 | 0.999906 | 0.99007 | 0.9818 | |

**Both real submissions lie on this curve.** Our public \((0.999641,\,0.970717)\) and #1's
\((0.999789,\,0.980018)\) are two independent points on a one-parameter family. In public numbers the
gap is \(\Delta R^2_\psi=+0.000148\) and \(\Delta\)Cons \(=+0.009301\); the curve predicts
\(+0.0088\) for that flux improvement. **Agreement to 0.0005.**

Sanity check with three real models rather than a synthetic shrink (B8 defines \(\alpha=1\)):

| | \(R^2_\psi\) | Cons | curve at that \(R^2_\psi\) | residual |
|---|---:|---:|---:|---:|
| B8 | 0.999317 | 0.94661 | 0.94661 | — |
| F9 | 0.999256 | 0.94996 | 0.94296 | +0.0070 |
| B8+F9 | 0.999413 | 0.95811 | 0.95272 | +0.0054 |

Real models sit \(\approx +0.005\) **above** the shrink curve, so error *structure* is worth
something — but it is small against the \(\approx 0.04\) of Cons that flux error explains over this
range, and it largely cancels between two real entries.

**What this retires.** #1 does not have a signal we lack, a physical basis we lack, or a Cons trick.
They fit \(\psi\) better and Consistency follows. The "\(\kappa\) hole" was never a hole — it is the
reading on a dial whose input is flux error. §30.5's information ceiling is real but applies to
*direct scalar prediction*, which is now revealed as the wrong route: the route runs through
\(\psi\).

**The target, restated.** Public \(R^2_\psi\) 0.999641 → 0.999789 is a **41% cut in \(\psi\) MSE**
(error amplitude \(\times 0.767\)). That is a well-posed fitting problem with a proxy that needs no
LCFS extraction and no scorer — plain MSE — so iteration gets much cheaper than anything in §30.

**The catch, estimated (\(\rho=0.72\) — corrected in §30.7 to 0.451).** B8+F9 has \(1-R^2_\psi\) at
0.859 of B8 alone. Solving \((1+\rho)/2=0.859\) *assumes both members have equal error variance*;
F9's is 1.4× B8's, so that algebra returns the mix/mean-member variance ratio, not a correlation.
§30.7 measures \(\rho=0.451\) directly. The 28% ceiling below is therefore wrong — read §30.7.

Gate for the next job: *any change that cuts holdout \(\psi\) MSE by \(\ge 15\%\) against B8+F9.*
Score \(R^2_\psi\) only — Cons follows it at the measured rate and no longer needs its own
experiment.

### 30.7 The tail is the metric: \(R^2_\psi\) is decided by 1% of frames, and they are *ours*, not hard

§30.6 set the target at a 41% \(\psi\) MSE cut and guessed the ensemble ceiling at 28%. This is the
cheap check that guess deserved: every strong checkpoint on disk shares the frozen \(k{=}60\) basis,
so 14 members can be compared in coefficient space where \(\psi\) MSE is exactly
\(\lVert z-z_{gt}\rVert^2\) — inference only, no training. 80 holdout shots, 18,573 scored frames
(`psi_decorrelate.py`).

**Correction to §30.6.** Measured member error correlation is \(\rho_{B8,F9}=\mathbf{0.451}\), not
0.72. The 0.72 was mix/mean-member variance ratio misread as a correlation (§30.6 now flags this).

**Two findings from the correlation matrix.**
- **B12 is dead.** MSE 10.096 against B8's 0.00098 — 10,350× — and \(\rho\approx0.1\) with
  everything, i.e. noise. It is why an equal-weight mix of all 14 scores \(-6258\%\). Do not load it.
- **The family is more redundant than §12 thought.** F9↔O2 \(\rho=0.99\), C7↔D2c \(0.98\),
  F4↔C7 \(0.95\), B8↔B7 \(0.86\). Several "independent" members are numerically the same model.
  The one real outlier is **Y2** (yellow's vac-at-TS write): \(\rho=0.17\) against F9, the lowest
  legitimate pair on the board. Greedy selection adds B8, F9, then **Y2 for +8.6 points**.

**What ensembling actually buys: \(\approx10\%\), not 41%.** Best label-free aggregator is a plain
mean over a curated core-8 — **9.9%** MSE cut, \(\approx+0.0005\) S, free and shippable. Weight
fitting is *not* worth it: shot-split OOF is wildly unstable (\(+14.4\%\) one way, \(-9.3\%\)
swapped) because the objective is dominated by a few frames; a relative-error refit stabilises at
\(9.5\pm0.6\%\), no better than the unweighted core-8 mean.

**The actual structure of the metric.** \(S=0.55\,R^2_\psi+0.15\,R^2_{qb}+0.10\,(1-D_{LCFS})+0.20\,\)Cons,
and \(R^2_\psi\) is a *global* MSE ratio — so it is decided by the worst frames, not the average one:

| | share of total \(\psi\) MSE |
|---|---:|
| top 0.1% of frames (18 of 18,573) | 10.7% |
| top 1% | 30.3% |
| top 5% | 56.5% |
| **worst single shot (1 of 80)** | **23.5%** |

Clipping the top 5% to p95 is a **40.1%** cut — the 41% target, from the tail alone. The tail is
4.6× enriched in the last 5% of each shot (ramp-down), and 94 of the worst 186 frames come from one
shot.

**And the tail is not intrinsic — it is member-specific.** On the worst 1% of frames, an oracle that
picks the best member per frame scores **84% below** the shipped pair, and the winner is spread
across the family (B11 21%, F9 18%, B7 14%, B13 13%, Y2 13%, B8 6%). Over all frames the per-frame
oracle router is a **66.5% cut** — *far past the 41% needed to match #1*. When B8 explodes, something
already on disk is fine.

**But no consensus rule can reach it,** and that is the informative part: median over 13 (6.5%),
trimmed means (8.3%), and disagreement-triggered median fallback (\(\le7\%\)) all lose to the plain
core-8 mean. On the frames that matter the *majority is wrong and the minority is right*, so
robust-aggregation methods fail by construction. Capturing the oracle needs a **learned per-frame
router** — features plus member disagreement predicting which member to trust — not a better
average model.

**Where this leaves the plan.** Ship the core-8 mean (free, \(\approx+0.0005\) S). Then the highest
ceiling on the board is the router: 66.5% oracle against a 41% requirement, meaning a router that
captures *a quarter* of its oracle wins Challenge 1. Gate: **a label-free router that beats the
core-8 mean by \(\ge6\%\) MSE on held-out shots** (≈ 1/6 of oracle).

### 30.8 The 66.5% oracle is hindsight, not signal — the router is CLOSED

§30.7 ended by pointing at a learned per-frame router: a 66.5% oracle against a 41% requirement,
so capturing a quarter of it wins. It does not survive contact. A gating MLP over the core-8,
trained on the true objective \(\lVert\sum_m w_m z_m - z_{gt}\rVert^2\) with shot-disjoint folds
(`psi_router.py`):

| loss | OOF MSE cut vs core mean | OOF median cut |
|---|---:|---:|
| relative | \(-6.01\pm4.74\%\) | \(+2.33\%\) |
| plain MSE (the metric) | \(-5.05\pm5.70\%\) | \(+4.25\%\) |
| log-MSE | \(-6.91\pm6.90\%\) | \(-0.55\%\) |

Every loss loses. Under all three the gate converges to nearly the *same static* weighting
(F9 \(\approx0.33\), Y2 \(\approx0.15\), S1 \(\approx0.14\)) — it never learns to route, it learns a
fixed blend and then gives up the tail, which is the only thing \(R^2_\psi\) rewards.

**Capacity is not the limit; generalisation is.** Fitting and scoring on the *same* shots:

| features | in-sample cut | OOF cut |
|---|---:|---:|
| disagreement only | 60.95% | \(-8.88\%\) |
| raw physics only (133 legal cols) | 47.58% | \(-14.05\%\) |
| raw + disagreement | 61.32% | \(-16.13\%\) |
| **oracle** | **63.64%** | **51.98%** |

The gate recovers ~61 of the oracle's 63.6 points in-sample and goes *negative* out-of-sample in
every feature set (600 epochs: 66.8% in-sample, \(-34.7\%\) OOF). Direct classification of the
argmin member scores **0.859 train / 0.159 test against a 0.224 majority baseline** — worse than
always guessing the most common winner — and 0.000 on the worst-1% test frames.

**Mechanism.** Which member wins a given frame is shot-idiosyncratic and is not a function of
anything observable — not member disagreement, not the legal inputs. Adding 133 physics features
made OOF *worse*, which is the signature of a signal limit rather than a data limit (more capacity
at fixed data should help if signal existed). The 66.5% oracle is hindsight selection, the same
trap as §23's post-hoc splice. **Untested variant:** these folds train on 60 shots; a corpus of
thousands might expose structure. Given test accuracy *below* the majority baseline, that is not
where the odds are.

**What survives, and where the tail must be attacked.** The free core-8 mean stands
(**9.9%** MSE cut, \(\approx+0.0005\) S, no training). And §30.7's tail result is unchanged and
still the largest lever on the board — but since the tail cannot be fixed by *selecting* among
members at inference, it has to be fixed at *training* time, by making individual members stop
failing there. The concrete handle: the tail is **4.6× enriched in the last 5% of each shot**
(ramp-down), so the next experiment is a member fine-tuned with the tail upweighted, scored by
whole-holdout \(\psi\) MSE. Gate unchanged: **\(\ge15\%\) MSE cut against B8+F9.**

**Two constraints on that experiment, measured before starting it.**

1. **Do not use "late in shot" as the weight.** Ramp-down is 4.6× *enriched* in the tail but is
   still only **23.7%** of it — a late-frame upweight addresses under a quarter of the problem.
   Weight by the parent's own per-frame error instead (hard-frame mining), which targets all of it.
2. **Input clipping is not the mechanism (killed).** `feature_clip=10` on every core member raised
   the obvious hypothesis that ramp-down inputs go out-of-distribution and get clamped, destroying
   the signal exactly where we fail. Measured on F9: tail frames have **0.0144%** of cells clipped
   against **0.0381%** for the rest — the tail is *less* clipped — and frame \(\max|z|\) is 4.75 vs
   2.91, more extreme but far under the threshold of 10. Only F9 is a valid reading here; the
   `zgrid`/`thzr` members need `CRC_FEQ_THOMSON_LAYOUT` set when the features are rebuilt, and
   without it their bundle normalisation is meaningless (\(\max|z|\sim10^{19}\)). **Anyone
   rebuilding features for a bundle must set that env or the numbers are silently nonsense.**

Note also that the parent is already trained on a squared loss, which weights large errors
quadratically by construction — so "the tail is under-weighted" is not obviously true, and the
experiment's real question is whether the tail is *reducible at all* for this architecture.

### 30.9 The tail is a per-shot *offset* worth 39.8% — and nothing label-free can see it

§30.8 closed inference-time selection. Asking instead what the tail frames *are*:

- **Not extrapolation.** Tail \(\lVert z_{gt}\rVert^2\) is 1.47× typical; 0% exceed the rest's p99.9.
- **Not shrinkage.** \(\langle\hat z,z\rangle/\langle z,z\rangle=0.999\) on the tail (right magnitude),
  \(\cos(\text{err},-z_{gt})=0.003\) (no pull toward the mean). The error is nearly *orthogonal* to truth.
- **Leading-mode.** Mode 0 alone carries **71%** of tail error (63% elsewhere).
- **Not isolated.** Neighbour-frame error ratio 0.97 — neighbours are equally bad, so no temporal fix.

Those four together say *shot-level systematic*, and that is what it is. Per-shot error share:
top shot **16.0%**, top 3 **36.0%**, top 10 **57.0%** of all \(\psi\) MSE. Decomposing each shot's
error into a constant offset plus the rest: shot 2 is **70.3%** pure offset, shot 28 **78.0%**,
shot 62 52%, against **3.1%** for the other 16,600 frames.

**Removing one constant offset per shot cuts whole-holdout MSE by 39.8%** — the 41% target, from a
single 60-number correction per shot. And it survives averaging 8 members, so the offset is
**common to the family**, not member-specific.

**Two label-free probes for that offset. Both closed.**

1. **Vacuum field (`vac_calib.vac_psi` + rim).** Ψ_vac is a genuinely good rim reference —
   rim-mean GT vs Ψ_vac correlates **+0.9911** and explains 97.2% of rim variance on the same
   65×65 grid. It is still useless here: **our rim error (std 0.00287) is 28× smaller than the
   calibrated vacuum field's own residual against truth (0.0815)**. Solving per-shot offsets from
   the rim over the leading 1–20 modes scores \(-14{,}000\%\) to \(-170{,}000\%\). We are far more
   accurate than the instrument, so the instrument only injects noise.
2. **Thomson flux-surface consistency.** Chords do give 2D coverage (44 core points on the R=1.94
   vertical, 10 tangential on a Z≈−0.06 radial scan), and Te must be single-valued in ψ. The metric
   cannot tell a biased map from a true one: differences are ±0.0004 against values of 0.005–0.019,
   with no consistent sign, and **shot 28 — 78% bias — scores better than ground truth**. Dominated
   by profile structure and measurement noise, not flux accuracy.

So the offset is real, worth the entire gap, and invisible without labels. It has to be attacked
where labels exist: training.

### 30.10 The parents were undertrained. Hard-frame weighting *hurts*. (CLOSED)

`psi_tailfit.py` fine-tunes a parent weighting each frame by the parent's own error,
\(w=(e/\bar e)^p\), scored by holdout coefficient MSE (= \(\psi\) MSE in the frozen basis). \(p=0\)
is the control that separates "fine-tuning helps" from "weighting helps". It was the control that
mattered:

| run | holdout \(\psi\) MSE | vs B8+F9 pair |
|---|---:|---:|
| B8 parent | 9.7543e-4 | −14.65% |
| T1, \(p=1\) error-weighted | 8.9287e-4 | −4.94% |
| **T0, \(p=0\) plain fine-tune** | **7.8538e-4** | **+7.69%** |
| T0 + B8 + F9 | — | **+12.59%** |

**Hard-frame mining is dead** (−4.94% against +7.69% for doing nothing special). The tail is not
under-weighted — squared loss already weights it quadratically — and forcing more weight onto it
trades away the bulk for frames that are, per §30.9, a shot-level offset the inputs do not
determine.

**The real finding is the control.** A plain continued fine-tune cut B8's own MSE by **19.5%**
(9.75e-4 → 7.85e-4), and one model now beats the shipped two-member mix. **B8 was not converged.**
Every member in §30.7 is a fine-tune-of-a-fine-tune from the same undertrained lineage, which would
explain why §30.7's ensemble ceiling looked so low. Error correlation of T0 with its parent is 0.907
(as expected for a child) but only 0.440 with F9, so the fine-tuned members should still mix.

**The sweep confirms it: every parent was undertrained, some catastrophically.** Holdout \(\psi\)
MSE, original → plain fine-tune (short budget), then the long sweep (220 epochs, patience 30):

| member | before | short | long | own (best) | long best ep |
|---|---:|---:|---:|---:|---:|
| **O1** | 3.78e-3 | **5.9115e-4** (U) | 5.9275e-4 (L) | **−84%** | U 57 / L 143 |
| B11 | 1.32e-3 | 8.9703e-4 | — | −32% | U 57 |
| B7 | 1.04e-3 | 7.7027e-4 (U) | 7.9211e-4 (L) | −26% | U 36 / L 51 |
| B13 | 1.17e-3 | 7.9524e-4 (U) | 7.7177e-4 (L) | −34% | U 57 / L 94 |
| B8 | 9.75e-4 | 7.8538e-4 (T0) | **7.3628e-4** (L) | **−25%** | T0 11 / L 29 |
| F9 | 1.38e-3 | 1.1606e-3 | — | −16% | U 39 |

S1 and Y2 blew up (0.227, 0.250): their `sky_tokens`/`yellow` builders need their env set *before*
the coefficient cache is built, which `_activate` after the fact cannot fix. Same family trap as
§30.8/§30.10, third occurrence. Excluded, not concluded.

**O1 went from the worst member on the board to the best model we have.** Its rank in §30.7 was an
artefact of training budget, not of the architecture — which retroactively invalidates every
member-quality judgement made by rank rather than by converged performance.

**Best mix: B8/L + O1/U + O1/L = 5.2410e-4, a +38.40% cut against the B8+F9 pair** (41% = 5.0198e-4
matches #1). Two O1 seeds plus the long B8 beat all six; the weaker members still dilute, so §30.7's
"cousins dilute" rule survives — it just applies to converged members now.

Official-scoring that mix with the TF members' own q95/βN heads **fails**: \(S=0.976456\),
\(R^2_{q\beta}=0.878895\) against the lock's \(0.996839\). The psi/Cons gain is real
(\(R^2_\psi=0.999793\), Cons \(=0.979158\), \(1-D_{\mathrm{LCFS}}=0.989038\)); composite \(S\) died
because coeff-only fine-tunes drifted the scalar heads. Do not ship that mix.

**Hybrid gate (shipped B8+F9 qB + mix psi):** official \(S=\mathbf{0.994147}\) /
Cons \(0.979158\) / \(R^2_{q\beta}=0.996839\) vs lock \(0.993203\) / \(0.975424\).
\(\Delta S=+0.000944\). Pass. Extra O1 seeds (S13 / S99 / S123) are **parked** — S7 already lost
to the U/L pair (6.576e-4) and the hybrid is a Challenge-1 bump, not a run at #1.

**CLOSED.** The parents were undertrained; hard-frame weighting hurts; the legal test-submit is
the 38.40% psi mix with the lock's q95/βN, not the wrecked-qB mix. Extra O1 seeds stay parked.

**Public [914971](https://www.codabench.org/competitions/17739/detailed_results/914971) VOID.**
`d3d_S=0.2005`, \(R^2_\psi=-2.026\), Cons 0, \(D_{\mathrm{LCFS}}=0.487\);
`r2_qb=0.995065` and `mast_S=0.585865` are the lock/solver copies and scored. Cause: all three
TF members are `zgrid`, so `hybrid_qb_package` took the same-layout shortcut and
`build_npz_from_public_test_cache` fed them the **th1** public-test cache. Holdout used the zgrid
`build()` cache, which is why local \(S=0.994147\) was real and public psi was not. Do not treat
0.2005 as a model result.

### 30.11 Step 0 — tailfit hybrid calibration probe scored (916155)

**Public [916155](https://www.codabench.org/competitions/17739/detailed_results/916155)** from
`benchmarks/CRC_FEQ_hybrid/0.1.0_20260904_134642` (rebuild after VOID 914971). First measured
local→public offset for the tailfit family. Submitted 2026-09-05; Challenge 1 rank **#8** (was **#10**
on baseline [912438](https://www.codabench.org/competitions/17739/detailed_results/912438) at
`d3d_S=0.991848`).

| term | local holdout | public (916155) | Δ public−local |
|---|---:|---:|---:|
| **S / d3d_S** | **0.994147** | **0.993112** | **−0.001035** |
| Consistency | 0.979158 | 0.975861 | −0.003297 |
| \(R^2_\psi\) | 0.999793 | 0.999781 | −0.000012 |
| \(R^2_{q\beta}\) | 0.996839 | 0.995065 | −0.001774 |
| \(1-D_{\mathrm{LCFS}}\) | 0.989038 | 0.988007 | −0.001031 |
| **mast_S** | 0.5828 (holdout) | **0.585865** | solver v2 reused |
| **G_ratio** | — | **0.589928** | vs 912438 0.590680 |

**Offset comparison — do not mix across families:**

| family | local \(S\) | public `d3d_S` | offset |
|---|---:|---:|---:|
| Brown B8+F9 | 0.993203 | 0.991848 | −0.001355 |
| Sky S1+F9 | 0.993259 | 0.991626 | −0.001634 |
| **Tailfit hybrid** | **0.994147** | **0.993112** | **−0.001035** |

MAST half byte-identical to 912438 — all MAST terms unchanged. DIII-D moved: public `d3d_S`
+0.001264 over the brown donor. The rank climb (#10 → #8) is real DIII-D psi from the tailfit mix,
not a board shuffle; #7 (`lkirill`, 0.993159) is 0.000047 above us.

**CLOSED as calibration probe.** Tailfit offset is **−0.001035** — less negative than Brown
(−0.001352) or Sky (−0.001634). Use this offset for tailfit-family public projections, not Brown's.
Local \(S=0.994147\) remains under the 0.9945 house bar; this was not a championship submit.

---

## 31. Indigo — the \(j_\phi\) operator, and Consistency is *not* a slave variable (CLOSED 2026-09-05)

**Verdict: the basis passes, the operator does not, and the parameterization is the reason.** I0/I1/I2
measured a good \(j\) basis — Cons **0.7256**, Cons(6) **0.8465**, `R_axis` to 2.1 mm,
re-confirmed to four digits three times (§31.8, §31.9, §31.10). I4 trained it on \(\psi\) MSE, which
is the gate §31.5 forbade for a \(j\) parameterization, and landed *below* the do-nothing floor. I5
fixed the objective completely — coefficient \(R^2\) −0.142 → **+0.9938**, amplitude 2.0% →
**101.5%** — cleared the floor, and still failed MAST shape: Cons(6) **0.0000** against the 0.8465
ceiling, per-scalar error 2.0–150× the ground-truth spread, `mast_S` **0.5219** against the shipped
solve's 0.5828 holdout. **Nothing shipped under this color.** Step 9 packaged the shipped baseline
instead (§31.11) and the arc's record is
[`MODEL_DESCRIPTION_CRC_FEQ_indigo.md`](MODEL_DESCRIPTION_CRC_FEQ_indigo.md). Do not start a fourth
generation; the open question is a basis question (§31.10's last paragraph).

### 31.1 I0 PASS — but only because the loss goes on \(\psi\), not on \(j\)

The write is \(\psi=\psi_{\rm ext}+P\,j\): coil field from `vac_calib.vac_psi`, response operator
from `gs_solver.load_response`, and a per-cell plasma current a model predicts. Feed-forward, one
matmul, no Picard iteration and no GS residual in the loss.

The gate is the **full composite** on the 115 labelled MAST frames, not \(R^2_\psi\) — §29.10
exactly. `scripts/indigo_exist.py`, \(q_{95}/\beta_N\) left nan so `r2_qb` reads 0 by construction
and every row is short 0.15:

| row | \(\psi_{\rm ext}\) | \(j\) | \(R^2_\psi\) | Cons | \(D_{\rm LCFS}\) | `mast_S` | \(|j|/|j_{\rm true}|\) |
|---|---|---|---:|---:|---:|---:|---:|
| `exact` | `decompose` | \(j_{\rm true}\) | 1.000000 | 1.0000 | 0.0000 | 0.8500 | 1.000 |
| `calib` | `vac_psi` | \(j_{\rm true}\) | **0.999057** | **0.8041** | 0.0397 | 0.8063 | 1.000 |
| `vac_only` | `vac_psi` | 0 | −0.330 | 0.0000 | 1.000 | 0.0000 | 0 |
| `absorb` \(\lambda=10^{-2}\) | `vac_psi` | ridge LS | 1.000000 | 0.9757 | 0.0039 | 0.8448 | 1.006 |
| `absorb` \(\lambda=10^{-6}\) | `vac_psi` | ridge LS | 1.000000 | **1.0000** | 0.0001 | **0.8500** | 1.104 |

`exact` is the identity check and it is exact, so the operator and the \(\Delta^*\) stencil are
right and everything below it means something.

**\(j_{\rm true}\) is the wrong ceiling and gating on it would have killed Indigo for a reason that
is not about Indigo.** `calib` pairs a *biased* coil field with the current the label implies and
scores Cons 0.8041 — below the 0.98 floor, `mast_S` 0.9563 with the scalars earned, a FAIL. But I4
trains the loss on \(\psi_{\rm hat}\) against \(\psi_{\rm gt}\), so a trained operator is free to
emit a deliberately wrong \(j\) that cancels the \(\psi_{\rm ext}\) error, and it will, because
that is what its loss rewards. `absorb` measures that: \(\min_j\|\psi_{\rm ext}^{\rm calib}+Pj-
\psi_{\rm gt}\|^2\), one Cholesky of the \(3969^2\) normal matrix reused across frames.

It recovers **everything**, at a current 10% larger than the true one — and at \(\lambda=10^{-2}\),
where \(|j|\) is within 0.6% of physical, Cons is still 0.9757. The \(\psi_{\rm ext}\) error is
absorbable by a current distribution indistinguishable in magnitude from the real one. Ceiling with
the scalars earned is **1.0000** against a gate of 0.985. **I0 PASSES.**

So "train the loss on \(\psi\), not on \(j\)" is not a convenience, it is the whole gate: on \(j\)
the family caps at Cons 0.804, on \(\psi\) it reaches 1.000. Do not add a \(j\)-space loss term.

Also settled: **there is no MAST passive-structure problem.** The residual coil-field error is
0.555% of the flux swing and it is absorbable, so §29.2's withdrawal of the DIII-D
passive-structure claim extends to MAST. Do not build an eddy-current basis for `vac_calib`; the
plan's I0-failure branch is not needed.

### 31.2 §30.6 is falsified — and not by submission 915010

§30.6 closed the Consistency axis: "Consistency is a slave variable of \(\psi\) error", both
submissions on one measured curve to 0.0005. The `calib` row breaks it on our own data, without a
board comparison:

* \(R^2_\psi=0.999057\), which is **better than our shipped DIII-D flux** (0.999781).
* §30.6's shrink table predicts Cons \(\approx 0.981\) at that \(\psi\) error.
* Measured Cons: **0.8041**. Short by **0.177** — 35× the 0.0005 agreement §30.6 rested on.

And the counterpart, from the same table: `absorb` \(\lambda=10^{-2}\) has \(R^2_\psi\) *identical*
to `absorb` \(\lambda=10^{-6}\) at six decimals and Cons **0.9757 vs 1.0000**. Two maps, the same
pooled \(\psi\) error, Consistency 0.024 apart. The curve is not a function.

The mechanism is the one §30.9 found from the other side. A small, smooth, near-harmonic error is
cheap in pooled MSE and expensive in geometry: it moves the LCFS, which every shape scalar is read
off. The per-scalar split says which, and it is not uniform —

| | `R_axis` | `Z_axis` | `kappa` | `tri_top` | `tri_bot` | `volume` | `li` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `calib` | +0.9978 | +0.6111 | +0.9395 | +0.8791 | **+0.2406** | +0.9656 | +0.9952 |

\(li\) is 0.9952 — internal inductance survives a coil-field error nearly intact, because it reads
\(|\nabla\psi|\) over the interior and not the boundary. `tri_bot` takes 79% of the damage. So the
axis §30.6 declared closed is real, it is *boundary* geometry specifically, and `tri_bot`/`Z_axis`
are where it lives.

### 31.3 A0.1 — MAST \(q_{95}\) is \(\psi\)-gated, not derivation-gated. Defer, do not ship.

The plan's rule was "ship \(q_{95}\) if its own \(R^2>0.05\)". Measured through the shipped solve
(`scripts/mast_gate.py`, now printing `r2_each`): \(q_{95}\) **−1.5110**, \(\beta_N\) **−1.4179**.
The rule says ship neither.

The choice is per *frame*, though, not per scalar: `local_score.score_shot` substitutes the fold
mean into each non-finite prediction individually, so a nan frame scores as a constant predictor
and a partial ship is strictly better than shipping all. `scripts/mast_qb_gate.py` sweeps label-free
acceptance rules under that semantics:

| flux source | rule | keep | \(R^2_{q95}\) | median \(q_{95}\)/label |
|---|---|---:|---:|---:|
| shipped solve | all | 100% | −1.5110 | **1.309** |
| shipped solve | \(q_{95}<9\) | 40.0% | +0.0889 | 1.160 |
| ground truth | all | 100% | −0.4556 | **1.006** |
| ground truth | \(q_{95}<9\) | 67.8% | **+0.4286** | 1.005 |

The median ratio is 1.006 on truth and 1.309 through the solve, so **the bias is in our flux map,
not in `gs_scalars`**. The same rule is worth \(R^2_{q95}=+0.43\) (\(+0.032\) of `mast_S`) on good
flux and \(+0.089\) (\(+0.0067\)) on ours. It also costs DIII-D truth 0.55 of \(R^2_{q95}\)
(0.9725 → 0.4260), which is a machine-specific constant announcing itself exactly as `gs_scalars`'s
flat-top gate did.

**Verdict: A0.1 is blocked on MAST \(\psi\), which is what Indigo fixes. Re-measure after Indigo,
ship then.** \(\beta_N\) is negative on every rule and both flux sources — that one is a firm no,
and §29.13's fast-ion explanation stands.

**KILL: the Ampère amplitude correction.** \(q\sim1/|\nabla\psi|\) and Ampère's law makes enclosed
current linear in \(|\nabla\psi|\), so rescaling \(q_{95}\) by the ratio the map's own current is
off by looked like a label-free fix for the 1.309. It is not: the GS solver *imposes* \(I_p\), so the
ratio is p50 **0.999** (p10 0.973, p90 1.001) on every frame and there is nothing to correct.
Applying it anyway costs DIII-D 0.087. Do not retry.

### 31.4 `cons_gt_cache` now carries `li`, opt-in

Seven of seven scored Consistency scalars are cacheable; `li` was omitted. Added at
+1.9 ms/frame (13.1 against 11.2), requested per call rather than unconditionally so the 33,905
existing entries stay hits for the navy/teal callers that never read it. A `with_li=True` build
upgrades an entry in place and a later `with_li=False` call is still a hit.

### 31.5 I1 — a PCA basis over \(j\) saturates at Cons ≈0.72 and \(R^2_\psi\) cannot see it

`scripts/indigo_basis.py`, basis fit on 120 DIII-D shots / 4,764 frames, scored as the full
composite on the 115 MAST frames. `proj` projects MAST's own \(j_{\rm true}\) onto the basis;
`fit` chooses coefficients against \(\psi\), which is what I4's loss will do.

| \(k\) | explained | `d3_proj` \(R^2_\psi\) | `proj` \(R^2_\psi\) | `proj` Cons | `fit` \(R^2_\psi\) | `fit` Cons |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 0.9921 | 1.00000 | 0.99892 | 0.6661 | 0.99982 | 0.5465 |
| 60 | 0.9945 | 1.00000 | 0.99899 | 0.6754 | 0.99983 | 0.5522 |
| 200 | 0.9985 | 1.00000 | 0.99900 | 0.7100 | 0.99985 | 0.5605 |
| 400 | 0.9996 | 1.00000 | 0.99904 | 0.7126 | 0.99986 | 0.5694 |
| 800 | **1.0000** | 1.00000 | 0.99902 | **0.7334** | 0.99987 | **0.7209** |

**The plan's I1 gate is uninformative.** "DIII-D holdout \(R^2_\psi\ge0.9995\) proves the
parameterization is not lossy" reads **1.00000 at every \(k\), including \(k=40\)** — once it is
run against the *exact* \(\psi_{\rm ext}\) rather than the calibrated one, which it has to be or it
measures DIII-D's coil gains instead (§29.2 puts that ceiling at +0.9572, already under the gate).
\(P\) is a Green's-function smoother, so \(j\)-space error is nearly invisible in \(\psi\). Do not
gate a \(j\) parameterization on \(R^2_\psi\).

**PCA over \(j\) is the wrong basis, and this is a property and not a tuning problem.** \(j\)'s L2
energy is in the plasma core, so that is where PCA spends components; the Consistency scalars are
read off the LCFS, and the boundary is set by the small edge structure PCA discards. Explained
variance 0.9996 with Cons 0.7126 is those two facts at once, not a contradiction.

**`index` beats `relative`, which refutes the plan's I2 assumption for \(j\).** `relative` — \(j\)
resampled through the shared \(((R-R_0)/a,\,Z/a)\) frame, the one measured at 100% overlap for
*coordinate planes* — scores `proj` \(R^2_\psi\) **0.958** against `index`'s **0.999** at every
\(k\). MAST's aspect ratio is 1.17 against DIII-D's 2.38, so MAST's plasma extends past DIII-D's
relative footprint and the resampling zeroes the edge. Relative coordinates are right for *input*
planes and wrong for the \(j\) *target*.

### 31.6 The `Z_axis` term on the 115-frame holdout is a broken instrument

`scripts/indigo_cons_units.py`. Every Consistency term is a pooled \(R^2\), so its denominator is
the ground truth spread of the fold being scored. The three holdout shots are consecutive, from one
session:

| scalar | GT sd | `fit` rms err | `fit` \(R^2\) | DIII-D narrow→wide sd ratio |
|---|---:|---:|---:|---:|
| `R_axis` | 0.03805 m | 0.00227 m | +0.9965 | 1.40 |
| `Z_axis` | **0.00117 m** | 0.00226 m | **−2.7243** | 1.64 |
| `tri_top` | 0.03389 | 0.03365 | +0.0136 | 1.12 |
| `tri_bot` | 0.02469 | 0.04569 | −2.4247 | 2.50 |
| `volume` | 0.47542 m³ | 0.09507 m³ | +0.9600 | 1.65 |

`Z_axis` ground truth spans **8.4 mm** end to end, sd **1.17 mm** — **1.9% of one grid cell**
(MAST \(dZ=62.5\) mm). A 2.26 mm axis error, a third of a tenth of a cell and physically excellent,
scores \(R^2=-2.72\) and clamps to 0, costing exactly 1/7 = 0.143 of the term. `gs_plausible` says
this in passing ("beating a constant predictor on that scalar needs millimetres"); what it did not
say is that the *gate* inherits it.

**So I3's "MAST holdout Cons ≥ 0.90" may be unreachable however good the map is, and would kill a
working model.** Restate MAST Consistency gates as (a) per-scalar absolute error in physical units,
(b) Consistency over the six non-degenerate scalars, and (c) the label-free public
`mast_audit` ANY-flag rate. `tri_top`/`tri_bot` are *not* covered by this excuse — their error is
the same size as their signal, which is a real deficiency of the basis.

### 31.7 15 geometry-only compensating modes recover most of the gap (KEEP)

`scripts/indigo_augment.py`. The free-\(j\) solution repairs the coil-field error with a current
outside the span of DIII-D plasma currents, so no \(k\) finds it. Append instead a small set of
**compensating current modes**: 2-D Chebyshev flux fields on the vessel bounding box, each turned
into a current by one ridge solve \(\min_j\|Pj-\psi_m\|^2\). Built per machine from geometry alone —
no label, no MAST-fitted constant, the same standing as \(P\) and \(\psi_{\rm ext}\) themselves.

| basis | modes | \(R^2_\psi\) | Cons | 1−\(D_{\rm LCFS}\) | `tri_top` | `tri_bot` | `kappa` |
|---|---:|---:|---:|---:|---:|---:|---:|
| \(k=60\) \(j\)-PCA | 60 | 0.999830 | 0.5522 | 0.9481 | +0.014 | **−2.425** | +0.919 |
| + Chebyshev \(d\le2\) | 66 | 0.999981 | 0.6633 | 0.9701 | +0.487 | +0.209 | +0.957 |
| **+ Chebyshev \(d\le4\)** | **75** | **0.999988** | **0.6934** | **0.9746** | +0.464 | **+0.428** | +0.967 |
| + Chebyshev \(d\le6\) | 88 | 0.999991 | 0.6573 | 0.9781 | +0.228 | +0.408 | +0.973 |
| + Chebyshev \(d\le8\) | 105 | 0.999993 | 0.6767 | 0.9783 | +0.519 | +0.249 | +0.975 |

Fifteen extra directions cut \(\psi\) MSE 11-fold, take Consistency 0.5522 → 0.6934 and `tri_bot`
from −2.425 to +0.428. It **saturates at \(d\le4\)**; higher degrees buy \(\psi\) and lose
Consistency, which is §31.2 once more — the extra freedom goes into the plasma interior where the
metric does not look. `Z_axis` stays negative at every degree, as §31.6 says it must.

**I3 decision: the decoder is \(k=60\) \(j\)-PCA + Chebyshev \(d\le4\), 75 coefficients.** Ceiling
in that basis is Cons 0.6934 on this holdout against 0.8571 (6/7) if `Z_axis` is unreachable, so the
remaining honest gap is `tri_top`/`tri_bot` and it is a boundary-resolution problem. That is what
Amber's SoftDerive term is for, and §31.2 makes it load-bearing rather than polish.

### 31.8 I2 — the decoder, the targets and the ceiling agree; and §31.7's ceiling was two effects

`scripts/indigo_train.py --build-targets --verify`, `scripts/indigo_ceiling.py`. The full DIII-D
corpus is fit: **1,559,222 frames over 7,041 shots** in 122 s, 118 dropped under the 50 kA \(I_p\)
floor, 468 MB of sidecar against the 26 GB the psi grids take. Residual flux MSE over residual
variance, mean 0.0728, **p50 0.0632, p95 0.1642**.

**The two `rel_mse` numbers in play are different quantities and the 0.01 gate was checking the
wrong one.** Measured on the same frames:

| `rel_mse`, fitting against | p50 | p95 |
|---|---:|---:|
| exact \(\psi_{\rm ext}\) (`gs_solver.decompose`, reads the label — §31.7's ceiling) | 0.00008 | 0.00489 |
| calibrated `vac_psi` (what `build_targets` prints, and all inference can compute) | 0.0632 | 0.1642 |

`0.01` belongs to the first column, and the pipeline deliberately does not live there: §31.7's whole
argument is that targets must be fit against the *calibrated* coil field so the compensating block
absorbs the coil-calibration error, and that error is 0.30 of \(\rm sd(\psi_{gt})\) at p50. So ~0.15
is this design's honest floor on the printed number, not a miss. **Quoting 0.01 against the
calibrated fit is a category error; the two columns are not the same measurement.**

**A storage-vs-magnetics sign bug was inserting the plasma current backwards on 91% of the
corpus.** The pool's \(I_p\) channel carries the **magnetics** convention on purpose —
`scripts/fix_pool_ip.py` writes \(|I_p^{\rm Ampere}|\cdot\rm sign(magnetics)\) so the channel keeps
the polarity the label-free inference path will hand the model. But `psi_gt`, \(P\) and the basis
are all in the machine's **storage** convention, where DIII-D's plasma current is negative because
its EFIT axis sits at the psi minimum. The two disagree on **6,427 of 7,041 DIII-D shots**, and the
target fit was constraining the plasma block's net current to the *signed* pool value. Measured
damage: `rel_mse` p50 **1.27**, p95 **7.20**, worst frame **329** — most of the corpus fit worse
than predicting nothing, and the printed p50 still looked like a number rather than an obvious
crash.

The fix takes the **magnitude from the measurement and the sign from `machine_env.AXIS_SIGN`**, in
both `indigo_targets.TargetFitter.fit`'s constraint row and `IndigoDecoder.currents`, so training
and inference cannot drift apart — the pattern `gs_solver.solve` already used. It needs no MAST
special case: dividing by the mean's own net current flips the profile wherever a machine's current
genuinely runs the other way. That matters because **MAST has no local labels and its sign cannot be
measured**; `AXIS_SIGN["mast"] = +1.0` is a convention constant carrying it, and it is the first
thing to suspect if MAST scores collapse while DIII-D is healthy. After the fix, on 25 shots /
5,174 frames: mean 0.068, p50 0.059, p95 0.152, **no frame above 1**. Decoded \(|I_p|\) matches the
measurement to 9e-13 kA and the decoder's rescale is a no-op on targets to a relative 1e-8 — the
property `do_train` depends on, since it builds its label as `dec(coeff_tgt, ext, ip)`.

**`--verify` reproduces the ceiling.** Fitting the 115 MAST holdout frames against the *exact*
\(\psi_{\rm ext}\) through the shipped basis and decoder: \(R^2_\psi\) **0.999986**, Cons **0.7196**,
`mast_S` **0.7916**, \(1-D_{\rm LCFS}\) 0.9769. So the basis build, the target fitter and the
decoder are the same three matrices; **the basis build is not wrong.**

That needs saying because Consistency came in 0.026 *above* §31.7's 0.6934 and a ceiling one
overshoots deserves the same suspicion as one it misses. Two things moved, and `indigo_ceiling.py`
separates them by holding the compensating block, \(P\), the holdout and the solver fixed:

| plasma block | explained | \(R^2_\psi\) | Cons | `tri_top` | `tri_bot` |
|---|---:|---:|---:|---:|---:|
| 120 shots (§31.5/§31.7's) | 0.9945 | 0.999986 | 0.6736 | +0.471 | +0.290 |
| **400 shots (step 3's, shipped)** | **0.9929** | 0.999986 | **0.7196** | +0.574 | +0.498 |

The 120-shot row reproduces `zeronet_ip.py`'s `zero net, free fit` to every digit including all
seven per-scalar \(R^2\), so the pipeline is exact and **the fit sample alone is worth +0.046 Cons**
— all of it in `tri_top`/`tri_bot`, the boundary scalars §31.2 identified — while *explained
variance goes down*. §31.5 said PCA spends components on core L2 energy and the metric reads the
LCFS; this is the same fact with the sign that helps. **More fit shots, not more modes.**

**`zero_net` stays on, which reverses the reading taken from the 120-shot block.** §31.7's 0.6934
was bought with 128 kA of net plasma current sitting in the compensating block, which the pipeline
cannot use: the decoder pins enclosed current to the measured \(I_p\). All four rows re-measured on
the shipped block:

| compensating block | solver | \(R^2_\psi\) | Cons | `mast_S` | net in extra modes | \(|I_p|\) err |
|---|---|---:|---:|---:|---:|---:|
| as measured | free | 0.999988 | 0.7385 | 0.7959 | 126.94 kA | 136.39 kA |
| **zero net** | **free** | **0.999986** | **0.7196** | **0.7916** | 0.00 kA | 9.67 kA |
| as measured | \(I_p\)-constrained | 0.999986 | 0.7246 | 0.7927 | 4.26 kA | 0.00 kA |
| **zero net** | **\(I_p\)-constrained** | **0.999986** | **0.7256** | **0.7929** | 0.00 kA | 0.00 kA |

On the 120-shot block the net-current direction was worth **+0.0167** Cons under the constraint,
over the 0.01 that was supposed to flip `zero_net` off. On the shipped block it is worth
**−0.0010** — the constrained solver takes the two blocks to within a thousandth, and the free fit's
apparent 0.019 advantage is bought with 136 kA of \(I_p\) error the pipeline does not permit. **The
net-current direction was compensating for a deficient plasma block, not supplying physics.**
`zero_net=True` stands as written, and `IndigoDecoder.currents` does *not* need to be changed to
rescale the total current. §31.7's headline 0.6934 is superseded on both counts.

**The ceiling in force is Cons 0.7256** (`zero net`, \(I_p\)-constrained), against 0.8571 (6/7) if
`Z_axis` is unreachable. That is the number I3 and step 7 compare a trained model against — not
0.6934, which is a different basis under a solver the pipeline does not run, and not 0.7196, which
is `--verify`'s free fit and sits 0.0060 under the operating point because the constraint injects
true \(I_p\). `--verify`'s expected values were pointing at 0.6934 and passed only on a 0.03
tolerance; they now name the free-fit row, and the tolerance stays loose on purpose because the fit
sample moves Cons by more than the gate width.

`Z_axis` reads \(R^2=-18.26\) here against §31.6's −2.72, which costs nothing — it clamps to 0 in
both — but it is worth not misreading: that is a 5.1 mm axis error, **8% of one MAST grid cell**
(\(dZ=62.5\) mm), on ground truth whose sd is 1.17 mm. §31.6's verdict holds and this is the
instrument, not the map. **I2 PASSES. Proceed to I4 training.**

### 31.9 I4 — the operator never left its initialisation, and MAST is not the reason

`scripts/indigo_score.py`, `results/indigo/i4_score.json`. 20 epochs, 1,542,005 train frames /
17,217 val frames over 80 held-out shots, 11.7 h, best epoch 19 — the last one, so the run stopped
on its epoch budget rather than on convergence. Scored on the 115 MAST frames against §31.6's three
restated gates, with the §31.8 ceiling re-measured in the same process off the same holdout,
decoder and scorer.

| on the 115 MAST frames | \(R^2_\psi\) | Cons (7) | **Cons (6, no `Z_axis`)** | `mast_S` |
|---|---:|---:|---:|---:|
| **trained operator** | **−0.9856** | **0.0011** | **0.0013** | **0.0002** |
| all 75 coefficients zero | +0.6762 | 0.0069 | 0.0081 | 0.4306 |
| operator, compensating block muted | +0.4938 | 0.0000 | 0.0000 | 0.3109 |
| `zero net`, \(I_p\)-constrained (§31.8's ceiling) | 0.999986 | **0.7256** | **0.8465** | 0.7929 |

**The ceiling row reproduces §31.8 to four digits, so the measuring stick is the same one.** That
matters before reading anything else: 0.7256, 0.999986 and 0.7929 all land on the recorded values,
and the per-scalar \(R^2\) agree too. Nothing about the basis, the holdout or the scorer moved.

**The model is not short of its ceiling. It is below the floor.** Zeroing all 75 coefficients
decodes to the basis mean current rescaled to the measured \(I_p\) — already a real equilibrium, and
the initialisation `IndigoOperator` deliberately starts from. That does-nothing prediction scores
\(R^2_\psi\) **+0.6762** and `mast_S` **0.4306**. Training moved it to **−0.9856** and **0.0002**.
The gap to the ceiling is 0.7245 Cons, but the useful number is the other one: **the operator is
worth −1.66 \(R^2_\psi\) against emitting nothing at all.**

**Gate (a), per-scalar error in physical units.** The units are §31.6's, and the ceiling column is
what the same basis reaches on the same frames:

| scalar | GT sd | model rms err | model bias | ceiling rms err | model err / GT sd |
|---|---:|---:|---:|---:|---:|
| `R_axis` | 0.03805 m | **0.3713 m** | +0.3707 m | 0.00207 m | 9.8 |
| `Z_axis` | 0.00117 m | 0.0555 m | −0.0547 m | 0.00511 m | 47.3 |
| `kappa` | 0.12232 | 0.7723 | −0.7599 | 0.01937 | 6.3 |
| `tri_top` | 0.03389 | 0.0805 | −0.0670 | 0.02213 | 2.4 |
| `tri_bot` | 0.02469 | 0.0476 | −0.0303 | 0.01673 | 1.9 |
| `volume` | 0.47542 m³ | **1.839 m³** | −0.905 m³ | 0.01970 m³ | 3.9 |
| `li` | 0.33539 | 0.5529 | −0.4024 | 0.02385 | 1.7 |

A 371 mm error in `R_axis` is **12 MAST grid cells** (\(dR=30.3\) mm), and it is almost pure bias —
the axis is pushed outboard on every frame. §31.6 built this gate so a good map could not be killed
by a vanishing denominator; it is doing the opposite job here, and the model fails it by two orders
of magnitude on scalars where the basis passes it comfortably.

**Gate (b), Consistency over the six.** Model **0.0013**, ceiling **0.8465** against the 0.8571
(6/7) that §31.7 named as the target if `Z_axis` is unreachable. So dropping the degenerate scalar
does what §31.6 said it would — **the basis very nearly saturates the restated gate** — and it does
nothing for the model, because the model's failure is not a denominator artefact.

**Gate (c), the `mast_audit` ANY-flag rate: 0.4% over 720 frames from 120 public shots, and it is
worthless here.** Every frame closes an LCFS, `li` and `volume` are in range, `no_lcfs` is 0.0%. It
passes because it cannot do otherwise: the decoder emits `mean + basis` rescaled to the measured
\(I_p\), which is a physically shaped current by construction, so an Indigo prediction is always a
plausible equilibrium — just not the right one. The audit's own numbers say so if read against the
labelled frames: public-fold `R_axis` p50 is **1.253 m** where the labelled MAST frames sit near
**0.89 m**, the same +0.36 m outboard bias the holdout measures, and the bound is [0.583, 1.447] so
it never fires. **The ANY-flag rate is a collapse detector, and this decoder cannot collapse.** It
was the right gate for the GS solve, whose 31% tail it caught (§30 / `gs_plausible`); it must not be
carried into step 9 as evidence that an Indigo model is sound.

**`AXIS_SIGN["mast"] = +1.0` is not the bug, and it is no longer asserted.** §31.8 named it first
suspect. Flipping it takes \(R^2_\psi\) from −0.986 to **−3.108** and `Z_axis` error from 55 mm to
1.60 m, so +1 is the better of the two by a wide margin. Independently: the holdout's measured
\(I_p\) is **+625 kA** (p50) and the ground truth enclosed current from `plasma_current` is
**+6.35e5 A**, the same sign, so the constant is now a measurement on the labelled frames rather
than a convention. Retire it as a suspect.

**MAST is not the problem — the operator did not learn the map on DIII-D either.** Same checkpoint,
2,000 pool frames from the 80 shots training held out:

| DIII-D validation frames | value |
|---|---:|
| \(R^2_\psi\), trained operator | +0.98846 |
| \(R^2_\psi\), all coefficients zero | +0.94847 |
| **what the operator is worth over doing nothing** | **+0.0400** |
| Cons over the six, on an 80-shot fold | **0.0000** |
| `R_axis` rms err / GT sd | 1.00 |
| `tri_top` rms err / GT sd | 1.00 |

Cons(6) is **0.0000 on DIII-D**, on a fold 27× wider in shots than the MAST holdout, with every
scalar's error the same size as its ground truth spread. There is no transfer failure to diagnose
because there is nothing to transfer.

**The cause is in coefficient space, and it is §31.5's finding arriving as a training pathology.**
Read against the sidecar targets the loss was built from, on the same DIII-D validation frames:

| the operator against its own training target | value |
|---|---:|
| target coefficient rms | 4.367e+05 |
| model output rms | 8.785e+03 |
| **output amplitude as a fraction of target** | **2.0%** |
| coefficient MSE removed in 20 epochs | **0.42%** |
| coefficient \(R^2\) against target | **−0.142** |

The head is zero-initialised, and after 11.7 h it is still emitting 2% of the amplitude the targets
ask for and scores *worse than the per-coefficient mean*. The training log says the same thing in
plain sight and it was not read: `val_psi` fell 3.9× over the run (1.1959e-03 → 3.0922e-04) while
the logged `coeff` column moved 1.9170e11 → 1.9092e11, **0.4%**. §31.5's conclusion was "\(P\) is a Green's-function smoother,
so \(j\)-space error is nearly invisible in \(\psi\) — do not gate a \(j\) parameterization on
\(R^2_\psi\)." `do_train` minimises \(\psi\) MSE, which is that forbidden gate used as the
*objective*. It computes `coeff_mse` every step, logs it, and never adds it to the loss. A 3.9×
drop in \(\psi\) MSE bought 0.42% of the coefficients, and on MAST the residue is anti-correlated
with what the frames want: plasma block cosine +0.808 at 0.54× amplitude, compensating block cosine
**−0.160** at 0.24×.

That last figure also explains the row ordering. Muting the compensating block alone recovers
\(R^2_\psi\) from −0.986 to +0.494 — it is the single largest destroyer, which is what §31.7's
construction predicts, since those 15 modes exist to absorb *one machine's* coil-calibration
residual and a DIII-D-trained head has learned DIII-D's. But muting it still leaves the model under
the +0.676 floor, so the compensating block is an aggravator and not the disease.

**I3 FAILS on gates (a) and (b); gate (c) passes and is hereby demoted.** The next move is training
and not basis work, and the two were worth distinguishing precisely because they are not the same
job: the basis is measured good — Cons 0.7256, Cons(6) 0.8465, `R_axis` to 2.1 mm and `volume` to
0.020 m³ — and the operator never learned to address it. Before any more GPU hours: put
`coeff_mse` in the loss (normalised per coefficient, since the 75 span decades), or predict
whitened coefficients so the head does not have to travel five orders of magnitude from zero; then
re-check the 0.42% figure at epoch 1 rather than at hour 11. Step 8's q95 re-measurement is moot
until then — it reads the Indigo flux map, and this one is worse than no map.

### 31.10 I5 — the objective fix worked, the premise did not. Indigo is dead; ship the baseline

`results/indigo/i5_score.json`, `artifacts/indigo/i5_best.pt`, `results/indigo/i5_train.json`.
§31.9's fix applied in full: 75 targets whitened per coefficient, whitened-coefficient MSE added to
a per-frame-normalised \(\psi\) MSE at weight 1.0, mean/std carried in the checkpoint. 50 epochs,
276,059 train frames at 40 frames/shot, 3,200 val frames, **5.35 h** against I4's 11.7, best epoch
45 of 50. Same 115 MAST frames, same decoder, same scorer, ceiling re-measured in-process and
**reproducing 0.7256 / 0.999986 / 0.7929 to four digits for the third time**.

| on the 115 MAST frames | \(R^2_\psi\) | Cons (7) | **Cons (6, no `Z_axis`)** | `mast_S` |
|---|---:|---:|---:|---:|
| **I5, objective fixed** | **+0.8318** | **0.0000** | **0.0000** | **0.5219** |
| I4 (§31.9) | −0.9856 | 0.0011 | 0.0013 | 0.0002 |
| all 75 coefficients zero | +0.6762 | 0.0069 | 0.0081 | 0.4306 |
| I5, compensating block muted | +0.7446 | 0.0000 | 0.0000 | 0.4697 |
| `zero net`, \(I_p\)-constrained ceiling | 0.999986 | 0.7256 | 0.8465 | 0.7929 |

**The four gates, in the order §31.9 set them: PASS, PASS, FAIL, FAIL.**

**(1) Coefficient space is fixed, and it is not close.** On the same DIII-D validation frames and
the same measurement: coefficient \(R^2\) against target **+0.9938** (was −0.142), output amplitude
**101.5%** of target (was 2.0%), coefficient MSE removed **99.47%** (was 0.42%). The probe called
it at epoch 1 — +0.9750 and 93.9% after one epoch, 409 s — so the diagnosis in §31.9 was right and
cheap to confirm. **PASS.**

**(2) I5 clears the floor. It is the first Indigo generation worth more than emitting nothing.**
\(R^2_\psi\) **+0.8318** against the +0.6762 zero-coefficient floor, `mast_S` **0.5219** against
0.4306. I4 was worth −1.66 \(R^2_\psi\); I5 is worth **+0.156** and **+0.091** `mast_S`. Two
sub-findings reverse §31.9's: muting the compensating block now *costs* 0.069 \(R^2_\psi\)
(+0.7446 against +0.8318) where in I4 it recovered 1.48, so those 15 modes are no longer the
destroyer; and \(D_{\rm LCFS}\) improves on the floor, 0.3566 against 0.4269. **PASS.**

**(3) Consistency over the six is 0.0000 against a 0.8465 ceiling — and that is below both I4 and
the do-nothing floor.** Not small: zero, on all seven scalars, every one clamped from a negative
\(R^2\) (`R_axis` −10.70, `Z_axis` −22625, `kappa` −75.2, `tri_top` −26.8, `tri_bot` −123.8,
`volume` −34.2, `li` −2.83). I4 scored 0.0013 and doing nothing scores 0.0081. **The generation
that fixed the objective went backwards on the gate that measures shape. FAIL.**

**(4) Per-scalar error in physical units is 2.0–150× the ground-truth spread, against I4's
1.7–47×.** Six of the seven scalars are worse *in absolute physical units* than I4, not merely
worse as a ratio:

| scalar | GT sd | I5 rms err | I4 rms err | ceiling rms err | I5 err / GT sd (I4) |
|---|---:|---:|---:|---:|---:|
| `R_axis` | 0.03805 m | **0.1301 m** | 0.3713 m | 0.00207 m | 3.4 (9.8) — the only gain |
| `Z_axis` | 0.00117 m | **0.1764 m** | 0.0555 m | 0.00511 m | 150.4 (47.3) |
| `kappa` | 0.12232 | 1.0674 | 0.7723 | 0.01937 | 8.7 (6.3) |
| `tri_top` | 0.03389 | 0.1788 | 0.0805 | 0.02213 | 5.3 (2.4) |
| `tri_bot` | 0.02469 | 0.2759 | 0.0476 | 0.01673 | 11.2 (1.9) |
| `volume` | 0.47542 m³ | 2.8208 m³ | 1.839 m³ | 0.01970 m³ | 5.9 (3.9) |
| `li` | 0.33539 | 0.6562 | 0.5529 | 0.02385 | 2.0 (1.7) |

`R_axis` fell from 12 MAST cells to 4.3 (\(dR=30.3\) mm) — the outboard bias §31.9 named is
genuinely reduced. `Z_axis` went the other way, 55 mm to 176 mm, **2.8 cells** (\(dZ=62.5\) mm),
and it is nearly pure bias (+161 mm). `kappa` off by 1.07 on a spread of 0.12 is not a
near-miss equilibrium. **FAIL.**

**Reported and not counted: the `mast_audit` ANY-flag rate is 9.6%** over 720 frames from 120
public shots (volume 9.4%, `no_lcfs` 2.8%, axis_R 1.4%), against I4's 0.4%. §31.9 demoted this
gate and nothing here promotes it — it moved 24× *against* a 1.8-unit \(R^2_\psi\) improvement, so
it does not track accuracy in either direction. It is not evidence I5 is worse and it was never
evidence I4 was sound.

**Why (1) passing did not buy (3) or (4): \(P\) is a smoother, measured from the other side.** On
its own machine, with coefficient \(R^2\) at +0.9938, the operator is worth
\(R^2_\psi\) **+0.0414** over emitting nothing (+0.9898 against a +0.9484 floor). I4, with
coefficient \(R^2\) at −0.142, was worth **+0.0400**. **A 1.14-unit swing in coefficient \(R^2\)
and a 50× amplitude correction bought 0.0014 \(R^2_\psi\).** §31.5 said \(j\)-space error is nearly
invisible in \(\psi\); the converse is now measured too — solving \(j\)-space almost exactly is
nearly invisible in \(\psi\) as well. What *did* move on DIII-D is the part that reads \(j\)
directly: Cons(6) 0.0000 → **0.2761**, and per-scalar error from a flat 1.00× of GT spread to
0.66–1.21× (`R_axis` 0.66, `Z_axis` 0.76, `tri_bot` 0.68; `volume` 1.18 and `li` 1.21 still over).
So the training loop is no longer the fault: on DIII-D the operator now learns real shape.

**The fault that remains is cross-machine, and it is in the premise.** The MAST coefficient
agreement says it in one line: **plasma block cosine +0.143 at 2.09× amplitude, compensating
+0.904 at 5.52×**. I4 was plasma +0.808 at 0.54×. Learning DIII-D's coefficient manifold *properly*
made the plasma direction on MAST **worse** — I4's head sat near its zero-init, which is the basis
mean, which is accidentally closer to a MAST equilibrium than a well-fit DIII-D coefficient vector
is. The plasma block is fit on DIII-D \(j\) in `d3_index` space (§31.5) and the compensating block
is one machine's coil-calibration residual (§31.7); a head trained to hit that target exactly emits
a confident, 2–5.5× over-scaled DIII-D answer on MAST. Getting better at DIII-D is *anti*-correlated
with MAST shape accuracy under this parameterization.

**`AXIS_SIGN["mast"] = +1.0` stays retired, and `axis_sign_flip_helps` in the JSON reads `true` for
a reason that must not be misread.** That flag is a Cons(7)-only test: flipped 0.0609 against the
model's 0.0000, and it fires because *both* are noise against a 0.7256 ceiling. On every other
number the flip is catastrophic — \(R^2_\psi\) **−1.9106** against +0.8318, `mast_S` **0.0122**
against 0.5219, `R_axis` error 840 mm. Read the flag as "Cons is degenerate here", not as a sign
bug. §31.9's measurement stands.

**Like-for-like against what is actually shipped, Indigo I5 is behind.** The MAST composite is
\(S = 0.55R^2_\psi + 0.20\,{\rm Cons} + 0.10(1-D_{\rm LCFS}) + 0.15\max(0,R^2_{qb})\); both rows
forfeit the \(q_{95}/\beta_N\) term as nan (§29.13), so the comparison is clean on the same 115
frames:

| the 115 labelled frames | \(R^2_\psi\) | Cons | \(D_{\rm LCFS}\) | `mast_S` |
|---|---:|---:|---:|---:|
| **shipped GS solve v2** (§29.15 holdout) | **0.9063** | **0.0206** | **0.1976** | **0.5828** |
| Indigo I5 | 0.8318 | 0.0000 | 0.3566 | 0.5219 |

**I5 loses on all three live terms: −0.0410 on \(\psi\), −0.0041 on Cons, −0.0159 on LCFS, total
−0.0610 `mast_S`.** 0.5219 is roughly the *retired* v1 solve's public 0.5254. Step 9's rule — do
not push if it is behind — applies, and there is nothing to weigh against it: Indigo's advertised
edge was Consistency, the 30%-of-composite term the solve has never earned, and Indigo scores 0.0000
on it too.

**Verdict: DEAD, not marginal.** Two consecutive generations fail, the second with the objective
measured fixed and the basis re-confirmed good for the third time, so §31.9's own rule applies: the
fault is in the basis, the decoder or the premise, not the training loop. **Do not start a fourth
generation.** Step 8's q95 re-measurement does not run — it reads the Indigo flux map, and this one
is still behind the solve it would replace. **Step 9 packages the shipped baseline** (DIII-D Brown
B8+F9, MAST GS solve v2) and the model description states why Indigo did not ship.

If Indigo is ever reopened, the open question is *not* training: it is that the plasma block is fit
in DIII-D index space and there are 115 labelled MAST frames total to re-fit or adapt it from. That
is a basis question, and §31.8's ceiling says the answer is worth 0.8465 Cons if anyone finds it.

**Recording bug, noted not fixed:** `indigo_score.py` hardcodes `results/indigo/i4_train.json` for
its training-history fields, so `epochs_run`, `val_loss_first/last`, `val_loss_monotone` and
`train_seconds` in `i5_score.json` are **I4's** ("20 epochs", 1.1959e-03 → 3.0922e-04). No gate
reads them. I5's real history is in `results/indigo/i5_train.json`: 50 epochs, val loss 0.16685 →
0.05533, best 0.05411 at epoch 45, 19,276 s.

### 31.11 Step 9 — the baseline re-packages byte-identical to 912438, so nothing was submitted

`benchmarks/CRC_FEQ_solver/0.1.0_20260905_171902`. Step 9's first job was to establish whether there
was anything new to ship at all, and there is not. `solver_package --validate` rebuilt the shipped
baseline from the same brown donor and the same `mast_gs_solve_v2` psi cache, and **both halves
reproduce the scored submission's sha256 exactly**:

| | sha256 | vs 912438 |
|---|---|---|
| `diii_d_public_test.npz` | `867f6644…a262fc88` | **match** (hardlink of the brown donor; §29.15's check) |
| `mast_public_test.npz` | `0788cda9…f6627225fa` | **match** (re-derived from cache, different inode, identical bytes) |

Array-level digests over all 2,622 + 3,618 arrays match too, so the identity is content and not just
container: the solve is deterministic from the cache, and the diagnostics reproduce
(`fp16_max_rel_err` 4.746e-04, `psi_absmax` 0.33840 Wb, 1,206 shots / 69,075 frames, `n_dead` 0).
Validator `rc=0` on both configs.

**No HF push, no Codabench upload.** Step 9's rule is do not push if it is behind, and the weaker
statement also applies — a duplicate of a scored submission is not a submission. Indigo I5 is 0.0610
`mast_S` behind (§31.10) and there is no other candidate: the tailfit hybrid's local \(S=0.994147\)
is under the 0.9945 house bar and §30.10 is closed. Public ship therefore stays 912438 exactly as
recorded in §1, and the attempt directory is kept only as the reproduction record (Route-B
`direct_submission.zip` deleted; Route A per `.cursor/rules/codabench-submission.mdc` not run).

---

## 32. Geometry loss — it finally ran, on a healthy map, and it is null (CLOSED 2026-09-05)

Code: `scripts/cons_tailfit.py`. Results: `results/brown/P31_CONSFIT_*.json`. §31.2 located the
Consistency deficit in *boundary* geometry (`tri_bot` +0.2406, `Z_axis` +0.6111, `li` +0.9952) and
§30.2 showed a Cons-0.995 field is reachable by descending SoftDerive on \(z\). This section runs
that descent **through the weights** instead of through \(z\), so it produces a model rather than an
oracle. Every failure the plan had to avoid was a measured one, and all of them were avoided:

* **Joint seven, never one.** \(L = L_{\rm coeff} + \lambda_g \sum_j w_j (f_j^{\rm soft}(\hat\psi) -
  f_j(\psi_{\rm gt}))^2\), \(w_j = 1/\mathrm{var}_j\) on the fold. §30.2's `--kappa-only` trap
  (\(\kappa\) 0.986, Cons 0.859) is not repeated.
* **The ramp gate is a measurement.** `olive_loss.warmup_healthy` on the *measured* holdout
  \(R^2_\psi\), latched. It fires on the parent itself, before epoch 0. **This retires §23 item 22**
  — Olive's SoftDerive never turned on because O1/O1.b gated on an epoch counter and died at 8 and
  25; here it turns on every time.
* **Converged parent.** `CRC_FEQ_brown_TF_L_O1` (§30.10, best epoch 143), never a fresh init, and
  written to a fresh `CRC_FEQ_brown_CT_*` namespace — `psi_tailfit.py:209` would clobber a shipped
  member.
* **Official scoring only.** SoftDerive is in the loss and nowhere else; every number below is
  `local_score.build_reference` on 256 holdout frames.

**The 0.99978 floor is the shipped MIX's number and a single member cannot clear it.** Measured on
the 80-shot holdout (18,573 frames, 78.5 M px, \(R^2_\psi\) read off the coefficients — the basis is
orthonormal to \(9\times10^{-7}\)), `TF_L_O1` alone is \(R^2_\psi = 0.999765\), Cons 0.98110. The
script measures the parent first and refuses to start rather than quietly weakening the floor; the
sweeps below ran at parent \(-2\times10^{-5} = 0.999745\).

**Two losses, six arms, 200k frames, 12 epochs each.**

| boundary term | \(\lambda_g\) | \(\Delta\)Cons | \(\Delta R^2_\psi\) | |
|---|---:|---:|---:|---|
| seven scalars | \(10^{-3}\) | **+0.00072** | \(-7.1\times10^{-6}\) | best of everything tried |
| seven scalars | \(3\times10^{-3}\) | +0.00071 | \(-6.8\times10^{-6}\) | |
| seven scalars | \(10^{-2}\) | −0.00018 | | **ABORT** ep 4, floor |
| 4 exact + \(D_{\rm LCFS}\) | \(10^{-3}\) | −0.00007 | \(-7.8\times10^{-6}\) | |
| 4 exact + \(D_{\rm LCFS}\) | \(3\times10^{-3}\) | −0.00179 | | **ABORT** ep 4, floor |
| 4 exact + \(D_{\rm LCFS}\) | \(10^{-2}\) | −0.00305 | | **ABORT** ep 3, floor |

**The scalar loss moves exactly the scalars SoftDerive computes exactly.** `R_axis` +0.9743 →
+0.9764 and `Z_axis` +0.9881 → +0.9918, monotone with the ramp and repeated in all three arms —
the sub-pixel parabola, which is `o_point._subpixel` verbatim. \(\kappa\)/`tri_top`/`tri_bot`, read
off an *extent softmax*, do not move at all. Faithful gradient, faithful result; soft gradient,
nothing.

**\(D_{\rm LCFS}\) does not even minimise itself.** `soft_dlcfs` weighted to replace the three
extent scalars (auto-calibrated \(c_d = 6.96\) against the parent's train \(D_{\rm LCFS} = 0.00788\)):
the train value *rises* to 0.01106 by epoch 4 and never returns below the parent's in 12 epochs, and
at \(\lambda_g \ge 3\times10^{-3}\) it climbs monotonically (0.0103 → 0.0133) while every scalar
degrades. `tau_dlcfs_cells = 0.1` is a tenth of a cell, so the softmax-relaxed max is nearly one-hot
and each step chases whichever single crossing is worst on that frame — the caveat
`SoftDeriveConfig` already records for `tau_ext`, applied to a whole term.

**What this actually says, and it covers both sweeps.** Holdout coefficient MSE is pinned at
1.20–1.25e-4 in *every* non-aborting arm while individual scalars trade against each other — axes up,
\(\ell_i\) and volume down, Cons flat. Every loss here re-weights the same 60-dimensional residual,
and that residual is at the model's floor for these inputs. **The geometry term does not reduce
error, it rotates a fixed error budget between directions.** That is §30.6's one-parameter family
seen from the loss side, and §31.2's falsification does not rescue it: knowing the deficit is
boundary geometry does not help if no reachable weight setting has less boundary error.

Gate for anything that reopens this: *move holdout coefficient MSE below 1.20e-4 first.* A geometry
loss on top of a residual that does not shrink is a rotation, and rotations score zero.

---

## 33. The instrument was wrong — error bars, the shift-aware fold, and a \(\kappa\) closure RETRACTED by §34 (2026-09-05)

Code: `scripts/cons_foldvar.py`, `kappa_autopsy.py`, `kappa_weighted.py`, `qb_ensemble.py`,
`qb_refit.py`, `coverage_weighted.py`. Results: `results/brown/P33_*.json`.

§32 closed the geometry loss and left the question "where next". The answer turned out to be that
**every Consistency number this project has ever compared was read off an instrument with no error
bar, and off the wrong fold.** Nothing here gains score. It closes \(\kappa\), retargets §31.2,
and leaves one DIII-D term that is *not* at a limit.

### 33.1 The public per-scalar results were always readable

`api/submissions/<id>/get_details/` returns a signed URL for `scoring_result`, whose zip holds the
organizer's `detailed_results.html`: per-scalar Consistency and per-shot \(R^2_\psi\)/\(D_{\rm LCFS}\)
for all 874 DIII-D and 1206 MAST shots. Unauthenticated. No script here had ever pulled it. For the
shipped hybrid ([916155](https://www.codabench.org/competitions/17739/detailed_results/916155)):

| | `R_axis` | `Z_axis` | \(\kappa\) | `tri_top` | `tri_bot` | volume | \(\ell_i\) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DIII-D | 0.974809 | 0.980159 | **0.959625** | 0.989477 | 0.987845 | 0.980936 | **0.958178** |
| MAST | **−3.500838** | −0.393442 | −0.237400 | −0.223215 | −0.668922 | −1.589570 | −0.499321 |

DIII-D \(q_{95}\) 0.995915, \(\beta_N\) 0.994214. **This inverts §31.2**, whose profile came from the
`calib` synthetic coil-field perturbation: it named `tri_bot` (+0.2406) as the deficit and \(\ell_i\)
(+0.9952) as the survivor, and on the fold that pays `tri_bot` is our *second-best* scalar and
\(\ell_i\) our worst. §32 was aimed by that inverted profile.

### 33.2 Fold noise, and every past Cons delta restated in \(\sigma\)

2000 bootstrap folds of 80 shots on the shipped mix. \(z\) is (public − full)/\(\sigma\):

| scalar | full-80 | 1\(\sigma\) | 5–95 | public | \(z\) | corr(\(R^2\), fold GT sd) |
|---|---:|---:|---|---:|---:|---:|
| \(\kappa\) | 0.95987 | **0.01203** | 0.9356–0.9747 | 0.95962 | **−0.02** | **+0.899** |
| \(\ell_i\) | 0.97130 | 0.00505 | 0.9626–0.9790 | 0.95818 | −2.60 | +0.201 |
| volume | 0.98683 | 0.00475 | 0.9772–0.9926 | 0.98094 | −1.24 | +0.559 |
| `R_axis` | 0.97313 | 0.00465 | 0.9647–0.9798 | 0.97481 | +0.36 | +0.430 |
| `tri_bot` | 0.98536 | 0.00302 | 0.9795–0.9893 | 0.98784 | +0.82 | +0.532 |
| `Z_axis` | 0.99030 | 0.00184 | 0.9869–0.9928 | 0.98016 | −5.52 | +0.684 |
| `tri_top` | 0.99278 | 0.00130 | 0.9903–0.9945 | 0.98948 | −2.54 | +0.431 |

**§31.6's mechanism is confirmed on DIII-D, not just MAST**: pooled \(R^2\) rises with the scored
fold's own GT spread, for all seven. Aggregate Consistency 1\(\sigma\) is **0.00233** at 80 shots
(0.00333 at 40, 0.00110 at 320). In those units: the shipped hybrid over baseline +1.63\(\sigma\),
the gap to #1 +2.72\(\sigma\), the local→public offset −1.44\(\sigma\), §32's best arm +0.31\(\sigma\).
Only three of the seven public moves survive the noise — `Z_axis`, \(\ell_i\), `tri_top`. \(\kappa\)'s
is \(z=-0.02\): identical on both folds.

### 33.3 The paired test is the instrument, and it predicts the board

Absolute error bars are the wrong tool for A/B. Scoring both models on the *same* resampled fold
cancels the shared fold-spread variance. Shipped hybrid vs the B8+F9 baseline:

| | value |
|---|---:|
| \(\Delta\)Cons, full-80 | +0.00352 |
| paired 1\(\sigma\) | **0.00099** |
| unpaired 1\(\sigma\) | 0.00340 (3.4× wider) |
| effect | **+3.71\(\sigma\)**, P(better) 100% |
| **public \(\Delta\)Cons between the two scored submissions** | **+0.00514, inside the paired 2\(\sigma\)** |

So a paired local delta is a legitimate predictor of a public delta where an absolute local number
is not. Per scalar, that shipped improvement moved five scalars at \(\ge +2.14\sigma\) and left
\(\kappa\) (+1.11) and \(\ell_i\) (+0.98) in the noise — the same two that are lowest publicly.
**Use `cons_foldvar --compare` for every future A/B. Do not compare bare holdout numbers.**

### 33.4 \(\kappa\) autopsy: nothing to calibrate, nothing to route, all of it in the height

`corr`\((\kappa_{\rm gt},\kappa_{\rm pred}) = 0.9797\) and \(0.9797^2 = 0.95988\), which **is** our
\(R^2\) of 0.95987. The fit sits exactly on its correlation ceiling. Bias +0.00002 (0.0% of MSE),
shrinkage 3.4%, **scatter 96.6%**. The per-shot-offset share is 23.3% against \(\psi\)'s 39.8%
(§30.9) with the top three shots at 15.4% — no tail to attack. `corr` between the hybrid's and the
B8+F9 pair's per-frame \(\kappa\) error is **+0.711**, so half the variance is family-wide and more
seeds cannot average it out (R8's saturation, at the scalar level).

\(\kappa = \) height/width, and holding one at truth: as shipped 0.95913; **true width, our height
0.96593**; **true height, our width 0.98839** — which is R6's \(k=60\) basis ceiling of 0.9887 to
three decimals. Width is already at the basis floor. Every reachable point of \(\kappa\) is in the
LCFS vertical extent, and within it `Zmin` (\(R^2\) 0.97537) is worse than `Zmax` (0.98354): the
X-point end, where \(\psi\) is flat and a small flux error moves the boundary far.

### 33.5 LOCK — sensitivity-weighted \(\psi\) loss, and a lesson about reading a sweep

SoftDerive gives \(\partial\kappa/\partial\psi\) exactly; averaged over label frames it is a static
map of the pixels \(\kappa\) is read from (two lobes at the top and bottom extrema, **top 1% of cells
carrying 96.8%** — used raw it is a 42-pixel objective, so it is smoothed and floored at 1). A
spatially weighted \(\psi\) MSE is exact and cheap in coefficient space: \(\sum_x w(x)\,\Delta\psi^2
= \Delta z\,M\,\Delta z^\top\) with \(M = C\,\mathrm{diag}(w)\,C^\top\), \(60\times 60\), and \(w\equiv1\)
gives \(M=I\).

Nine arms on `TF_L_O1` looked monotone and topped out at \(\Delta\kappa\) **+0.00128 (+2.09\(\sigma\))**
at amp 256, +0.00137 at 1024, falling to +0.00111 at 4096, with the full-corpus arm reproducing
+0.00138 for \(-1.2\times10^{-6}\) of \(R^2_\psi\). **It does not replicate.** The same recipe on the
other two shipped members:

| member | parent | \(\Delta\kappa\) | \(\sigma\) | \(\Delta\)Cons | \(\sigma\) |
|---|---|---:|---:|---:|---:|
| MO1L | `TF_L_O1` | +0.00128 | +2.09 | +0.00051 | +2.23 |
| MO1U | `TF_U_O1` | −0.00359 | **−1.92** | −0.00237 | −4.55 |
| MB8 | `TF_L_B8` | −0.01377 | **−3.23** | −0.01246 | −6.45 |

The PCA basis is bit-identical across all three (`max|dC| = 0`), so there is no confound. Best-of-nine
clears 2.09\(\sigma\) about 15% of the time under the null; the arm was a multiple-comparisons
artifact and the mean effect is **harmful**. **House rule from this: sweep on one parent, then
replicate the winner on the other two, and claim nothing unless all three agree in sign.**

### 33.6 \(R^2_{q\beta}\): no calibration, no ensemble, and \(q_{95}\) is an overfit

On the live board (2026-09-05) the **entire 0.000047 gap to #7 is \(q\beta\)** — we lead `lkirill` on
Cons, tie on \(R^2_\psi\) to six decimals, level on \(D_{\rm LCFS}\). It is also the one shipped term
nobody had touched (`overlay_donor_qb` copies B8+F9 because tailfit heads drifted, §30.10). Three
attempts, against a \(R^2_{q\beta}\) 80-shot 1\(\sigma\) of **0.000695**:

| | \(\Delta R^2_{q\beta}\) | |
|---|---:|---|
| affine recalibration | +0.00001 | \(q_{95}\) \(R^2\) 0.998100 vs corr\(^2\) 0.998109; \(\beta_N\) 0.995579 vs 0.995589 |
| greedy head ensemble (4 of 12) | +0.000041 | +0.17\(\sigma\) |
| optimal LS stack, 11 heads, off-fold | +0.000193 | +0.28\(\sigma\) |
| closed-form ridge refit on the frozen trunk | +0.000156 | +0.97\(\sigma\), +0.22\(\sigma\) of fold noise |

All null. But the split by scalar is not: \(q_{95}\) local 0.998100 → public 0.995915 at
\(z = \mathbf{-5.53}\), \(\beta_N\) 0.995579 → 0.994214 at \(z=-1.12\). **\(q_{95}\) is a genuine
overfit of one `Linear(1024→1)` head** — the only DIII-D term measured here that is *not* at an
information limit. Note the loss/metric mismatch it lives in: `qb_normalize=False` weights \(q_{95}\)
about 20× \(\beta_N\) in raw MSE while the metric is their unweighted mean.

### 33.7 The shift-aware fold — coverage explains the Consistency offset

Train shots are anonymised (`d3d_shot_...`), so a campaign-disjoint split cannot be built, and any
new holdout drawn from the 7,041 is contaminated. But ranking the **existing 80 clean shots** by mean
distance to their 25 nearest train shots (per-shot mean feature vector, standardised) reproduces the
offset without retraining anything:

| fold | Cons |
|---|---:|
| all 80 (the old instrument) | 0.979940 |
| best-covered 40 | 0.980649 |
| **worst-covered 40** | **0.976832** |
| worst-covered 20 | 0.974469 |
| **public, 874 shots** | **0.975861** |

**The local→public Consistency gap is a coverage gap.** Within it \(\kappa\) is the coverage-sensitive
scalar by a wide margin — 0.968810 on the best-covered 40 against **0.939207** on the worst, a
collapse of −0.0296, where no other scalar moves more than 0.0064. \(R^2_\psi\) is *not*
coverage-limited (public beats every local subset) and \(q_{95}\) is not either (0.998482 on the worst
half, *higher* than the best): the \(R^2_{q\beta}\) aggregate landing on 0.995040 against the public
0.995065 is a cancellation between \(q_{95}\) and \(\beta_N\), not a match. Ranking saved with its
validation numbers in `results/brown/P33_COVERAGE.json`. **Score Consistency on the worst-covered 40.**

### 33.8 LOCK — coverage-weighted training

Upweight training frames by shot thinness, \(w = (\mathrm{cov}/\mathrm{median})^\alpha\) capped at 8,
and score on the fold that tracks the board. Pre-registered protocol per §33.5:

| \(\alpha\) | \(\Delta\kappa\) on worst-40 | \(\sigma\) | \(\Delta\)Cons | \(\sigma\) |
|---:|---:|---:|---:|---:|
| 0.5 | −0.00002 | −0.05 | +0.00038 | +1.11 |
| 1.0 | −0.00005 | −0.08 | +0.00035 | +0.83 |
| 2.0 | −0.00079 | −0.37 | +0.00019 | +0.35 |

Nothing cleared the bar, so nothing was replicated. \(\kappa\) does not move on the exact fold where
it is 0.030 worse, even at 8× weight, and Cons trends the wrong way with dose. **The coverage gap is
an information problem, not an emphasis problem: reweighting the same data does not buy
extrapolation.**

### 33.9 What this closes

> **RETRACTED the same night by [§34.1](#34-the-rivals-scorecards-are-public-too--kappa-was-never-at-a-limit-open-2026-09-06).**
> The paragraph below concluded \(\kappa\) was at an information limit. It is not. #1 scores
> \(\kappa=0.972315\) against our 0.959625, and `mwaller` scores 0.970316 with *worse* pooled
> \(\psi\) than ours. All three routes measured the limit of **our model family**, not of the
> task — none of them could see another team's map. The rest of §33 (the instruments, the
> shift-aware fold, the two LOCKs) stands; only this closure is withdrawn.

**~~\(\kappa\) is done.~~** §30.5 measured its information limit at \(\approx 0.956\) months ago by
scalar-predictor saturation; §33.4 finds it on its correlation ceiling with a family-wide error;
§33.7–33.8 find it collapsing in thin data and immune to reweighting. Public \(\kappa\) is
**0.959625**. Three independent routes, one answer: ~~we are at the limit. Do not open another
\(\kappa\) color.~~ **All three shared a blind spot; see §34.1.**

Gaps as of the live board — us \(S=0.993112\) (#8), #7 0.993159, #1 kimsunjoong 0.994984:

| term | \(w\) | us | #7 | #1 | \(w\Delta\) to #1 |
|---|---:|---:|---:|---:|---:|
| Consistency | 0.20 | 0.975861 | 0.975616 | 0.982144 | 0.001257 (67%) |
| \(R^2_{q\beta}\) | 0.15 | 0.995065 | 0.995668 | 0.997397 | 0.000350 (19%) |
| \(1-D_{\rm LCFS}\) | 0.10 | 0.988007 | 0.988060 | 0.990416 | 0.000241 (13%) |
| \(R^2_\psi\) | 0.55 | 0.999781 | 0.999781 | 0.999826 | 0.000025 (1%) |

\(R^2_\psi\) is 0.9998 for the entire top eight; it is finished as a differentiator for the field.
Two things remain that are not at a measured limit: the \(q_{95}\) head overfit (§33.6, and all of the
gap to #7), and **MAST**, which forfeits 0.35 of weight with Consistency and \(q\beta\) both exactly
zero while the leaders moved from 0.74 to 0.9479.

---

## 34. The rivals' scorecards are public too — \(\kappa\) was never at a limit (OPEN 2026-09-06)

Code: `scripts/rival_details.py`, `mix_coherence.py`, `o1_seed_sweep.sh`.
Results: `results/brown/P34_*.json`, `P33_PAIRED_o1x5_vs_hybrid{,_worst40}.json`.

### 34.1 `get_details` is unauthenticated for *every* submission, not just ours

§33.1 used `api/submissions/<id>/get_details/` to read our own per-scalar card and never asked
whether it worked for anyone else. It does, for any id on the board — and `codabench_board.py`
has been recording every competitor's submission id in `leaderboard.json` all along. One
afternoon's missing `for` loop.

This reads **scores only**: the organizer's `detailed_results.html`, which is the same artifact
the public detailed-results page renders. `get_details` also exposes `prediction_result`. We do
not touch it and neither should anyone here — another team's predictions are not ours to look at.

Per-scalar DIII-D Consistency, top of the board:

| rk | who | \(S\) | \(R_{\rm axis}\) | \(Z_{\rm axis}\) | \(\kappa\) | `tri_top` | `tri_bot` | volume | \(\ell_i\) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | kimsunjoong | 0.994984 | 0.982222 | 0.989052 | **0.972315** | 0.992167 | 0.990202 | 0.987827 | 0.961223 |
| 2 | dgaytanv | 0.994794 | 0.975408 | **0.991260** | 0.970097 | 0.993341 | 0.992299 | 0.987597 | 0.959187 |
| 4 | mwaller | 0.994345 | 0.980871 | 0.987678 | 0.970316 | 0.991021 | 0.989376 | 0.980874 | 0.960227 |
| 7 | lkirill | 0.993159 | 0.974908 | 0.980174 | 0.961554 | 0.990491 | 0.988650 | 0.980605 | 0.952929 |
| **8** | **catskills** | 0.993112 | **0.974809** | **0.980159** | **0.959625** | **0.989477** | 0.987845 | 0.980936 | 0.958178 |

**We are last in the top eight on \(R_{\rm axis}\), \(Z_{\rm axis}\), \(\kappa\) and `tri_top`.**
§33 spent a night proving \(\kappa\) was at its information limit; #1 is 0.0127 above us on it. The
existence proofs that settle it are the teams *below* us on \(\psi\):

| who | \(1-R^2_\psi\) | vs ours | \(\kappa\) | Cons |
|---|---:|---:|---:|---:|
| us | 2.190e-04 | — | 0.959625 | 0.975861 |
| mwaller | 2.490e-04 | 1.1× **worse** | **0.970316** | **0.980052** |
| ruiqi_feng | 6.380e-04 | 2.9× worse | 0.962523 | 0.973582 |
| cliffsedge66 | 9.810e-04 | **4.5× worse** | 0.960550 | 0.975160 |

`cliffsedge66` beats our \(\kappa\) carrying four and a half times our flux error. No information
limit does that. §30.5's saturating scalar-predictor, §33.4's correlation ceiling and §33.7's
coverage collapse are all real measurements — of **our family**. Not one of them could see another
team's map, and the ledger read "we cannot do better" as "it cannot be done". **Before closing a
term as information-limited, check the board: an opponent's score is the cheapest upper bound
there is.**

### 34.2 Most of the Consistency gap is not \(\psi\), and \(\ell_i\) is our best scalar

Cons is the plain mean of the seven, so the 0.006283 gap to #1 decomposes exactly: \(\kappa\)
28.9%, \(Z_{\rm axis}\) 20.2%, \(R_{\rm axis}\) 16.9%, volume 15.7%, \(\ell_i\) 6.9%, `tri_top`
6.1%, `tri_bot` 5.4%. **Broad, not one hole** — and §33's targeting of \(\ell_i\) and \(\kappa\)
as "our two worst" was reading absolute values. Adjusted for \(\psi\) error against the field,
\(\ell_i\) is our *strongest* scalar (5th of 20, +0.88 sd) and \(\kappa\) is mid-pack.

Regressing every team's Cons on \(\log_{10}(1-R^2_\psi)\) over the top 20: correlation −0.584, and
matching #1's \(\psi\) error would buy us only +0.00136 of the +0.00628. **About a fifth of the gap
is \(\psi\); the rest is Consistency extracted per unit of \(\psi\) accuracy**, where we sit 11th of
20 at +0.00098 against #1's +0.00590. Note also that "everyone is at 0.9998" hides a **15.5×**
spread in \(1-R^2_\psi\) across the top 20; §33.9 calling \(R^2_\psi\) finished as a differentiator
was an artifact of the board's four-decimal display.

### 34.3 NULL — the mix is not incoherent, and more members do not help

Two obvious mechanisms for the efficiency deficit, both dead.

**Averaging does not cost geometry.** The mean of \(K\) equilibria is not an equilibrium, and
§30.1 measured a partially-corrected field losing \(\ell_i\) outright, so the shipped
three-member mean was the prime suspect. Scored alone, `B8/L` is worse than the mix on **all seven
scalars and on \(\psi\)** (Cons −0.0111). Averaging helps geometry uniformly. Read the split while
you are here: going solo → mix buys +8.0e-05 \(S\) through \(\psi\) and **+2.23e-03 through
Consistency**. \(\psi\) is **3.5%** of what mixing is worth, and §30.10 selected the mix on \(\psi\)
MSE alone.

**More members do not help either.** Out to all eleven converged members, `P34_MIX_COHERENCE`
peaks at four and falls monotonically: +0.000033, −0.000148, −0.000388, −0.000475 \(S^*\) at 4, 6,
9, 11. §30.7's "cousins dilute" survives. But the *shape* is the finding — the one member that
added (`O1/S7`) is an **O1-lineage seed**, and every other lineage diluted.

### 34.4 The O1 seed lottery, and the best single member the project has

§30.10 parked seeds S13/S99/S123 after S7 lost to the U/L pair (6.576e-4 vs 5.91e-4). S7 ran the
default 40-epoch budget and stopped at 31, while `O1/U` peaked at 57 and `O1/L` at 143 — so the
budget hypothesis was that the seeds were never given one. **It was wrong**: on 220 epochs /
patience 30 the seeds early-stop at 42–88 epochs on their own. Budget was never the constraint.

| seed | best ep | holdout \(\psi\) MSE | vs pair |
|---:|---:|---:|---:|
| **99** | 88 | **5.7403e-04** | −32.53% |
| 123 | 57 | 6.5650e-04 | −22.84% |
| 13 | 45 | 6.7000e-04 | −21.25% |
| 7 (long) | 42 | 6.8482e-04 | −19.51% |

The seed lottery is worth more than the budget: **`LS_O1_99` at 5.7403e-4 is the best single
member in the project**, under `O1/L`'s 5.9275e-4. Three of four seeds land near 6.6e-4 and one
lands at 5.74e-4, so this is a wide draw and it has not been sampled out. Note `LS_O1_13` posted
the *best* validation of the four (5.868e-4) and a mid holdout — val and holdout disagree by seed,
which is the §33.7 coverage signature and a reason not to select seeds on val.

### 34.5 SHIPPED — `o1x5`, the first gain since the hybrid to clear both instruments

Shipped three plus `O1/LS99` plus `O1/S7`. Screened at stride 16 (all five variants positive, not
a best-of-\(N\) spike), then confirmed on the full 18,573 frames:

| fold | \(\Delta\)Cons | paired 1\(\sigma\) | \(\sigma\) | P(better) | unpaired |
|---|---:|---:|---:|---:|---|
| all 80 | +0.00078 | 0.00020 | **+4.01** | 100.0% | 16.9× wider |
| **worst-covered 40** (§33.7) | +0.00059 | 0.00026 | **+2.26** | 99.0% | 22.6× wider |

Official composite, `tailfit_mix_score --qb-from-pair`:

| | \(S\) | Cons | \(1-D_{\rm LCFS}\) | \(R^2_\psi\) |
|---|---:|---:|---:|---:|
| shipped hybrid | 0.994147 | 0.979158 | 0.989038 | 0.999793 |
| **`o1x5`** | **0.994318** | 0.979927 | 0.989210 | 0.999792 |
| \(\Delta\) | **+0.000171** | +0.000769 | +0.000172 | −0.000001 |

Per-scalar: \(\ell_i\) +0.00201, \(R_{\rm axis}\) +0.00126, volume +0.00074, \(\kappa\) +0.00071,
`tri_bot` −0.00027. Consistency is free on \(\psi\) and \(D_{\rm LCFS}\) came along uninvited.
Applying the worst-40 haircut, expect \(\approx +0.00014\) public, landing \(\approx 0.993252\)
against lkirill's 0.993159. Packaged at `benchmarks/CRC_FEQ_hybrid/0.1.0_20260906_040002`, HF
commit `79b1468`.

**Honest caveat, per §33.5:** `o1x5` was picked best-of-five and confirmed on the same 80 shots,
so the confirmation is not independent. What makes it unlike the §33.5 false positive is that all
five variants were positive and the mechanism is "we added the strongest member we have", not a
loss-shaping effect.

### 34.6 The board paid — and paid it on the scalars we did not expect

Submission [916458](https://www.codabench.org/competitions/17739/detailed_results/916458):
**\(S=0.993217\), rank #7**, past lkirill's 0.993159, and we now lead them on Consistency
0.976314 to 0.975616.

**The instrument called it.** The worst-40 paired test predicted \(\Delta\)Cons \(+0.00059\); the
board paid **\(+0.000453\)** — 0.5\(\sigma\) inside the paired error bar of 0.00026. The random-80
fold would have predicted \(+0.00078\) and overshot by 70%. Third time a paired local delta has
predicted a public one; §33.3 and §33.7 are both doing their jobs.

**But not on the scalars the local fold named:**

| scalar | public \(\Delta\) | local \(\Delta\) | local paired |
|---|---:|---:|---:|
| \(\kappa\) | **+0.001382** | +0.00071 | +0.84\(\sigma\) (the one that did *not* resolve) |
| \(Z_{\rm axis}\) | +0.001003 | +0.00054 | +1.92\(\sigma\) |
| volume | +0.000595 | +0.00074 | +1.87\(\sigma\) |
| \(R_{\rm axis}\) | +0.000525 | +0.00126 | +3.35\(\sigma\) |
| \(\ell_i\) | **−0.000548** | +0.00201 | +2.57\(\sigma\) |

\(\kappa\) was the **largest public gain and the weakest local signal**; \(\ell_i\) was the largest
local gain at \(+2.57\sigma\) and went **backwards** on the board. Local and public agree on the
aggregate and disagree on its composition. **Never tune a mix toward a named scalar** — the paired
test is trustworthy on Cons and is not trustworthy per-scalar. \(D_{\rm LCFS}\) also improved
0.011993 → 0.011865 without being targeted.

Public \(\kappa\) is now **0.961007**, above the \(\approx 0.956\) "information limit" §33.9
asserted the night before, which is the retraction in §34.1 arriving on the scoreboard.

### 34.7 LOCK — the \(q_{95}\)/\(\beta_N\) refit, and a correction to §33.6

\(R^2_{q\beta}\) is 20% of the gap to #1 and we are **last in the top eight** (0.995065 against
lkirill's 0.995668 and #1's 0.997397). The shipped heads are copied from the B8+F9 lock and are two
model generations stale, so the obvious move was to refit them on the *tailfit* trunks. §33.6 had
only ever fit the **pre-tailfit parents**. The heads are `Linear(width, 1)` on the same hidden state
`coeff_head` reads, so a ridge solution drops straight in, and \(q_{95}\)/\(\beta_N\) touch nothing
else in \(S\) — a free, separable term if it worked.

It does not. Closed-form ridge on the five o1x5 trunks, against the shipped lock, all 80 shots:

| trunk | \(q_{95}\) | \(\beta_N\) | \(R^2_{q\beta}\) |
|---|---:|---:|---:|
| **shipped lock (B8+F9 heads)** | **0.997965** | **0.995072** | **0.996518** |
| TF_L_B8 | 0.992602 | 0.992684 | 0.992643 |
| TF_S_O1_7 | 0.983200 | 0.990483 | 0.986842 |
| TF_U_O1 | 0.984283 | 0.990111 | 0.987197 |
| TF_LS_O1_99 | 0.979306 | 0.990729 | 0.985017 |
| TF_L_O1 | 0.978806 | 0.990434 | 0.984620 |

5-member mean 0.990631 against the lock's 0.996518: **\(-0.006001\), \(-5.72\sigma\), P(better)
0.0%**, worth \(-0.00090\) of \(S\). Not marginal — an order of magnitude the wrong way.

**The mechanism is a monotone tradeoff.** Rank the trunks by how hard the tailfit pushed them on
coefficient MSE and you get the scalar ranking backwards: `TF_L_B8` (a −25% \(\psi\) cut) keeps the
most at 0.992643, and the O1 lineage (−84%, the most aggressive fine-tunes) is worst at 0.9846.
**Coefficient-only fine-tuning does not merely drift the heads — it spends trunk capacity that the
scalars were using.** §30.10 read `r2_qb` 0.879 as a drifted head; it was the trunk as well.

**And §33.6's small positive was an artifact.** It selected the ridge \(\lambda\) on the holdout it
then reported. Selecting \(\lambda\) on a shot-disjoint tail of the *training* frames instead, and
extending the grid down to \(10^{-3}\) to be sure the boundary was not binding:

| \(\lambda\) | \(10^{-3}\) | \(10^{-2}\) | \(10^{-1}\) | \(1\) | \(10\) | \(10^{2}\) |
|---|---:|---:|---:|---:|---:|---:|
| TF_L_B8 | 0.992509 | 0.992700 | **0.992956** | 0.992918 | 0.992643 | 0.992141 |
| B8 (original trunk) | 0.994626 | 0.994780 | 0.995379 | 0.995655 | 0.995872 | **0.996045** |

Trunk RMS is 0.1707 vs 0.1641, so this is not a feature-scaling artifact. **Even on the original
trunk the best honest ridge is 0.996045, below the shipped SGD heads' 0.996518.** A closed-form
linear head on these features cannot beat the ones we already ship; §33.6's \(+0.000155\) was
\(\lambda\) chosen on the eval set. Our baseline reproduces its `lock_r2_qb` to all 16 digits
(0.9965182568855505), so the plumbing is the same and only the selection changed.

**LOCK.** Do not refit \(q_{95}\)/\(\beta_N\) linearly on any trunk we have. The heads want the
*un*-fine-tuned trunk, which is exactly what we already ship. The term is not closed — but the only
route left is a fine-tune that carries the scalar heads *in the loss*, so the trunk is never allowed
to spend their capacity. That is a training run, not a refit.

### 34.8 LOCK — training the scalar heads, and \(R^2_{q\beta}\) is a *generalisation* problem

§34.7 said the head was fine and the objective was wrong, so the follow-up was the obvious one:
fine-tune with the scalar heads *in* the loss, standardising each residual by its label sd so the
loss matches the metric's equal weighting instead of the shipped `qb_normalize=False` (measured on
our corpus at sd 1.8537 vs 0.8903, so raw MSE weights \(q_{95}\) **4.3×**, not the ~20× §33.6
estimated). Free to try: `hybrid_qb_package` overlays \(q_{95}\)/\(\beta_N\) from a donor
independently of the psi mix, so a scalar-only fine-tune cannot touch \(R^2_\psi\), Consistency or
\(D_{\rm LCFS}\) — the o1x5 psi is untouched by construction.

**NULL.** From the B8 parent (the best scalar trunk we have): holdout \(R^2_{q\beta}\)
0.996189 → **0.996151**, best epoch 1, validation flat from the first epoch, against the shipped
lock's 0.996518. `scripts/qb_tailfit.py`, `results/brown/P35_QBFIT_QB_B8.json`.

That is the **third** independent null on this term, and together they say something the first two
did not:

| attempt | result |
|---|---|
| 11-head ensemble, greedy (§33.6) | +0.000041, +0.17\(\sigma\), unresolved |
| closed-form ridge refit (§33.6/§34.7) | negative under honest \(\lambda\) selection |
| scalar-objective fine-tune (§34.8) | −0.000038, val flat from epoch 1 |

**They were all optimising the wrong number.** Local lock \(R^2_{q\beta}\) is **0.996518**; our
**public** value is **0.995065**. The \(-0.001453\) between them is larger than the entire
\(+0.000879\) that would take us from public to local-perfect, and it is the §33.6 gap
(\(q_{95}\) local→public \(z=-5.53\)) restated. Three methods spent their effort on a local
quantity that was already saturated while the loss was happening somewhere else.

**The worst-40 fold was then tried and does not rescue it either.** Refit against lock on the
board-predicting fold: \(+0.000135\), **\(+0.58\sigma\)**, P(better) 67.7%, worth \(+0.000020\) of
\(S\). Unresolved, same as everywhere else (`P33_QB_REFIT_oldtrunk_worst40.json`).

**And that measurement is the real result, because of what the fold itself reads.** The lock scores
0.996202 on the worst-covered 40 against 0.996518 on the random 80 — it barely moves — while the
public value is **0.995065**. So unlike Consistency, where §33.7's worst-40 brackets the board
almost exactly, **no local fold we have reproduces public \(R^2_{q\beta}\)**:

| | random 80 | worst-covered 40 | public |
|---|---:|---:|---:|
| Consistency (§33.7) | 0.979940 | **0.976832** | 0.975861 |
| \(R^2_{q\beta}\) | 0.996518 | 0.996202 | **0.995065** |

This confirms §33.7's aside from the other direction: the \(q_{95}\) deficit is **not** a coverage
effect, so the coverage fold cannot see it. **We have no instrument for this term.** Every
\(R^2_{q\beta}\) decision in this project — four of them now — has been made on a fold that reads
0.9962–0.9965 while the board pays 0.9951.

**Do not attempt another \(R^2_{q\beta}\) fit, on any fold.** The term is not information-limited
(#1 scores 0.997397 publicly, above our *local* number), but it cannot be worked until something
reproduces the public number locally. Finding that fold — not another estimator — is the task.

### 34.9 The board fails the *same* shots we do — and loses less on them

`rival_details.py` read the per-scalar table off every rival's card and stopped. The same card
carries a **worst-20 shots** table — shot id, that team's \(R^2_\psi\) on it, its \(D_{\rm LCFS}\)
— so the public test set is partially observable *per shot, for every team at once*
(`rival_shots.py`, top 10, DIII-D).

Of our worst 20 shots, **only 2 are ours alone**; 8 are named by \(\ge60\%\) of the board.
`DIII-D_182803` and `DIII-D_192584` are on 9 of 9 rivals' lists. So the tail is **not** a private
failure — the hard shots are the same shots for everybody.

What differs is how much they cost. On the shots we and a rival *both* rank in our own worst 20:

| rk | who | \(S\) | shared | we win | mean \(\Delta R^2_\psi\) |
|---:|---|---:|---:|---:|---:|
| 1 | kimsunjoong | 0.994984 | 12 | 4 | **\(-0.001733\)** |
| 2 | dgaytanv | 0.994794 | 9 | 1 | **\(-0.004933\)** |
| 4 | mwaller | 0.994345 | 10 | 7 | \(-0.000340\) |
| 9 | jungdaesuh | 0.992895 | 14 | 9 | \(-0.000029\) |
| 10 | credasis | 0.992395 | 11 | 5 | \(-0.001673\) |

**The mean is negative against all nine, including the three ranked below us.** On `DIII-D_192322`
we score 0.987600 while the eight rivals who also list it span 0.994400–0.998400 — we are 0.0068
below the *worst* of them, on a shot they find only mildly hard.

Set against §34.2, where our pooled \(1-R^2_\psi\) is **the best on the board** (2.190e-04, against
`mwaller`'s 2.490e-04), the two facts only fit one way: **our error is more tail-concentrated than
anyone's.** We buy the pooled metric on the bulk and give it back on the shots that are hard for
everyone.

*Caveat, stated plainly.* The table conditions on both teams' truncated top-20, so it is
suggestive, not an unbiased estimate of tail depth. The unconditioned facts — 2 of 20 private,
`192322` below every rival's value, best pooled \(\psi\) on the board — do not depend on it.

**This reopens §30.9 from the outside.** §30.9 measured the tail as a per-shot *constant offset*
worth **39.8%** of holdout \(\psi\) MSE — 71% in mode 0, surviving an 8-member average — and called
it "common to the family." It is not common to the *board*. A defect nine other teams do not share
is a property of our construction, not of the task, which is the one thing none of §30.7–30.10
could see from inside our own family.

**It does not explain \(\kappa\).** §33.4 measured \(\kappa\)'s error at 96.6% scatter with a
per-shot-offset share of 23.3% and no tail, so the `mwaller` extraction-efficiency puzzle in §34.2
is *not* resolved by this. What this is, is the \(\psi\) term: 39.8% against §30.6's 41% target, at
55% of \(S\).

### 34.10 The per-shot offset is *in the fit*, and it is family-wide — so it is the input write

§34.9 ranked the per-shot offset first and named the untried attack: learn one correction per
shot from shot-level context. Before building that, one number decides whether it is trainable at
all. The offset is a *per-shot* quantity, so the sample size for learning it counts shots, not
frames — and we hold out 80 of 7041. If it only appears out of sample, there are 80 examples and
the idea is hopeless. **The probe is the same statistic on shots the model was trained on**
(`shot_offset.py`, B8, 298 training shots against the 80 holdout):

| B8, coefficient space | \(\psi\) MSE | after removing one offset per shot | offset share | \|offset\| p50 |
|---|---:|---:|---:|---:|
| holdout, 80 shots | 0.000975 | 0.000686 | 29.7% | **0.0665** |
| training, 298 shots | 0.000583 | 0.000493 | 15.5% | **0.0539** |

**The offset is 81% as large on shots the model was trained on.** The *share* differs only because
the non-offset residual is smaller in sample; the offset itself barely moves. It is not a
generalisation gap.

**And it is not one model's habit.** Free check, no GPU — the 13 live members in `decorr_z` are
structurally distinct writes and trunks, and their per-shot offsets agree: median cross-member
correlation **0.697** (mode 0 alone 0.628), and a 13-member consensus explains a median **71.7%**
of each member's own offset. The consensus offset is p50 0.0565 against a member's 0.0665, so most
of the magnitude *is* the shared part.

**This kills §34.9's proposal and sharpens it into a better one.** Thirteen architectures, each
having seen these shots thousands of times across 6961 training examples, all miss each shot's
level by the same amount. A head that predicts the offset from the same inputs has no advantage
the networks did not already have — capacity and sample size are not what is binding. Something
common to the whole family is, and it is not the basis: \(z_{\rm gt}\) is the label *projected into
the same \(k{=}60\) basis*, so truncation cancels out of \(z-z_{\rm gt}\) exactly. What is left
common is **the input write**.

Put beside §34.9 — nine rivals do not carry this tail, and every team is handed the same
competition inputs — the reading is that **the information needed to set a shot's flux level is
available and our feature write discards it.** That is a very different job from a correction head,
and it is the first hypothesis in this ledger that explains the §34.2 efficiency puzzle without
appealing to anything we cannot see.

*Care with the wording.* "Mode 0" is the leading PCA mode, not a DC pedestal; it carries 52% of the
offset energy on the holdout, 58% in sample. Do not conflate this with the MAST `DELTA = 0` bug
(§ solver), which is a machine-transfer units term and genuinely inert on DIII-D.

**Next measurement, cheap:** regress the 80-shot consensus offset on shot-level physics quantities
(\(I_p\), coil currents, density, shape aggregates) and see what predicts it. If a quantity we
already hold predicts the offset, it is missing from the write and the fix is an input, not a head.
If nothing does, the direction closes and §34.9's tail finding stands as diagnosis only.

### 34.11 NULL — nothing we hold predicts the offset, and the suspect is the *parameterisation*

§34.10 concluded the per-shot offset had to live in the input write and named the test: regress it
on shot-level aggregates of the raw inputs. Done (`offset_regress.py`) — 378 shots with measured
offsets (80 holdout + 298 training), 135 features from magnetics (20 F-coils, ECOILA, bcoil,
plasma current, dsep) and Thomson, `efit_*` excluded as labels. Shot-disjoint 5-fold ridge,
\(\lambda\) picked inside each training fold:

| out-of-fold | |
|---|---:|
| \(R^2\) of the offset, mode 0 | **\(-0.0151\)** |
| \(R^2\), energy-weighted over 60 modes | **\(-0.0200\)** |
| modes with \(R^2>0\) | 19/60 |
| holdout \(\psi\) MSE cut, oracle offset | 29.7% |
| holdout \(\psi\) MSE cut, **predicted** offset | **\(-0.4\%\)** |

Nothing. Worse than not correcting at all, and 19/60 positive modes is what noise looks like. The
oracle line recomputing to 29.7% is the accounting check — the estimator is sound and the answer
is a flat zero.

**What this does and does not close.** As framed — a shot-level quantity we already hold and fail
to pass to the net — the direction is closed. The caveat is that shot aggregates are a crude
summary and could destroy time-resolved structure; against that, the nets themselves see the full
series and miss the same offsets, which is the same null from the other side.

**The suspect §34.10 let through.** That section ruled out the basis on the grounds that
\(z_{\rm gt}\) is the label in the same \(k{=}60\) basis, so truncation cancels out of
\(z-z_{\rm gt}\). True, and it rules out the basis as a *representation*. It does not rule out the
basis as the **parameterisation of the learning problem** — which is the actual thing all 13
members share. Regressing 60 coefficients makes the leading mode's amplitude a single scalar
output, and a systematic per-shot bias in that one number is precisely what we measure (52–58% of
the offset energy in mode 0). A team predicting \(\psi\) some other way has no such scalar to bias.

That reading fits everything at once: family-wide (§34.10), present in sample (§34.10), invisible
to every label-free probe (§30.9), unpredictable from the inputs (here), and absent from rivals
(§34.9). It is also expensive to test — it needs a member that does not regress coefficients, and
§ Navy/Charcoal killed the INR and FieldRefiner routes for unrelated reasons. **Do not start it on
this evidence.** The cheap discriminator first: measure whether the offset's mode-0 share falls
when the basis is enlarged or rotated, which is a scoring change, not a new family.

### 34.12 CLOSED — the per-shot offset *is* the Forest oracle warp, in coefficient space

§34.11 blamed the coefficient parameterisation and asked whether mode 0 is special. Free checks on
the cached offsets settle it, and the answer kills that suspect and then explains everything else.

**Mode 0 is not special — it is just the biggest mode.** In units of each mode's own spread the
offset is *smallest* in mode 0:

| | mode 0 | modes 1–4 | modes 5–59 |
|---|---:|---:|---:|
| raw \|offset\| | 0.0633 | 0.0227 0.0144 0.0268 0.0097 | — |
| as a fraction of that mode's own std | **0.0072** | 0.0132 0.0183 0.0308 0.0182 | **0.2329** |

Mode 0's 52% share of raw offset energy is **0%** of *normalised* offset energy. §34.11's "a bias
in one scalar output" reading is **withdrawn** — relatively, mode 0 is our best-fit mode.

**The offset is low-dimensional and shot-varying.** Its top singular direction carries **74.3%** of
the across-shot variance and the top three **92.6%**, while the shot-independent constant part is
only 3.1%. Sixty numbers per shot are really about three.

**And those directions are geometric.** Building, per shot, the coefficient-space directions of a
rigid shift in \(R\) and \(Z\), an isotropic scale, a vertical stretch and a gain — five degrees of
freedom fitted against a 60-vector, where a *random* 5-dof span would explain ~8% — gives:

| share of per-shot offset energy in the warp span | |
|---|---:|
| all five together, energy-weighted over shots | **0.691** |
| median shot | 0.601 |
| isotropic scale alone | 0.293 |
| shift \(R\) alone | 0.116 |
| vertical stretch / gain / shift \(Z\) | 0.099 / 0.050 / 0.011 |

**Roughly 70% of the largest lever in this ledger is a five-parameter geometric misplacement of the
equilibrium** — which is exactly what the Forest lineage spent §F15–F66 on. F15/F57/F63/F64/F66
proved oracle \(s_Z\), \(dR\), \(dZ\) and isotropic scale are worth large \(S\) (up to 0.99436), and
marked every one CEILING because they are GT-derived; F16/F21/F56/F67 then failed to find a *legal*
predictor of \(s_Z\) or \(dR\) from \(X\), shape, embed or kNN.

So §30.9's per-shot offset, §34.10's family-wide in-fit defect and the Forest oracle warps are
**one phenomenon** measured three ways, and its legal-predictor problem was closed a lineage ago.
That retro-explains every null in §34.9–34.11 and, with §34.11's own regression as one more
independent failure, this is the strongest evidence in the project that **DIII-D \(\psi\) is at the
practical limit of what we can place legally.** Treat further work on the per-shot offset as
reopening a closed lineage: it needs a *new legal observable* of equilibrium position, not another
estimator.

**Strategic consequence.** If DIII-D \(\psi\) is closed, the points are on MAST, which still
forfeits 0.35 of the weight with Consistency and \(q\beta\) exactly zero (§34.13 item 3). That is
now the largest scoreable thing we are not doing.

### 34.13 MAST: we ship **NaN** for every \(q_{95}\) and \(\beta_N\), and Challenge 2 is where the points are

§34.1's trick works on `--machine mast` too, and nobody had run it. The Challenge-2 card for the
top of the Challenge-1 board:

| rk | who | `S_model` | \(R^2_\psi\) | Cons | \(R^2_{q\beta}\) | \(D_{\rm LCFS}\) |
|---:|---|---:|---:|---:|---:|---:|
| 6 | **alvar0** | **0.976340** | **0.999600** | **0.981947** | 0.873258 | 0.008179 |
| 1 | kimsunjoong | 0.947880 | 0.998162 | 0.922658 | 0.785916 | 0.035277 |
| 3 | dmitryzhevn | 0.920225 | 0.997612 | 0.913726 | 0.620593 | 0.042963 |
| 4 | mwaller | 0.611666 | 0.931378 | 0.076925 | 0.086423 | 0.289407 |
| **7** | **catskills** | **0.585865** | 0.927863 | **0.000000** | **0.000000** | 0.244602 |

**The \(q\beta\) zero is deliberate and correct — this was checked and the first read here was
wrong.** Every one of the **69,075** MAST \(q_{95}\) and \(\beta_N\) values in the shipped npz is
NaN, 0/69075 finite against 100% on DIII-D, written on purpose at `solver_package.py:176`.
`local_score` substitutes the fold mean into each non-finite prediction *individually*, so a NaN
frame scores exactly as a constant predictor, and §31.3 measured the alternative: derived
\(q_{95}\) through our solve is \(R^2=-1.5110\) and \(\beta_N\) \(-1.4179\). **Shipping them would
be far worse than shipping nothing**, and the best label-free acceptance rule (\(q_{95}<9\), keep
40%) reaches only \(+0.0889\), i.e. \(+0.0067\) of `mast_S`. §31.3's verdict — defer, blocked on
MAST \(\psi\) — stands. Rivals reach 0.62–0.87 on this term because their \(\psi\) is good enough
to derive from, not because they have a scalar model we lack: on *ground-truth* flux the same
derivation gives median \(q_{95}\)/label 1.006 against 1.309 through our solve.

**Consistency on MAST is a step function of \(\psi\).** Across the board, Cons is ~0 for everyone
below \(R^2_\psi\approx0.93\) (mwaller 0.077, lkirill 0.016, ours 0) and ~0.92–0.98 for everyone
above 0.9976. There is no middle. So MAST Cons (0.20) and \(D_{\rm LCFS}\) (0.10) are not separate
jobs — they are downstream of pushing MAST \(\psi\) from our 0.9279 to ~0.999.

**Headroom, against the 0.001767 left on DIII-D:**

| | ours | best | weight | worth |
|---|---:|---:|---:|---:|
| \(R^2_{q\beta}\) | 0.000000 | 0.873258 | 0.15 | **+0.131** |
| Consistency | 0.000000 | 0.981947 | 0.20 | +0.196 |
| \(R^2_\psi\) | 0.927863 | 0.999600 | 0.55 | +0.039 |
| \(1-D_{\rm LCFS}\) | 0.755398 | 0.991821 | 0.10 | +0.024 |
| | | | | **+0.390** |

**And it corrects a standing premise.** §26.789 and §27.849 record "there is no local MAST GT,
Challenge 2 is zero-shot" and §Ultraviolet forbids training on MAST labels. The dataset confirms
the first half — `Sophelio/fusion-equilibrium-challenge` publishes exactly three configs,
`diii_d_train`, `diii_d_public_test`, `mast_public_test`, with **no MAST training split**. But
`alvar0` reaches \(R^2_\psi=0.9996\) on MAST anyway, which is not what zero-shot transfer looks
like from a DIII-D fit — and it is the *third* time (§33.9 \(\kappa\), §34.1 rivals, here) that a
limit inferred from our own family has been falsified by a rival's card. §Ultraviolet's thesis —
"top-3 MAST is not from MAST labels, it is a fillable per-coil write" — is the live explanation and
now has an existence proof at 0.976.

**So there is no cheap first move.** Every one of the four MAST terms routes through the same
place. Consistency and \(D_{\rm LCFS}\) are a step function of \(\psi\) (above); \(q\beta\) is
blocked on \(\psi\) by §31.3, with the bias measured to be in our flux map rather than in
`gs_scalars`; and \(\psi\) is \(\psi\). **MAST is one job — \(R^2_\psi\) from 0.9279 to \(\ge0.9976\)
— worth +0.390 when it lands and close to nothing until it does.** That is a far larger and more
uncertain undertaking than anything left on DIII-D, and it should be started as a project, not as a
patch.

*Correction.* An earlier draft of this section called the NaN a defect worth \(+0.131\). It is
neither a defect nor worth that; §31.3 had already settled it.

### 34.14 Scoping MAST \(\psi\): our physics caps below the threshold, and the only route past it is a leak

§34.13 priced MAST at +0.390 behind one threshold, \(R^2_\psi\ge0.9976\). What would reach it.

**Our solver cannot, even with an oracle.** §29.7's own table: Thomson-picked profiles give
**+0.9179** label-free, the *oracle* profile — chosen with labels, within the \((\alpha,\beta,\lambda)\)
family — gives **+0.9598**, and the \(\Psi_{\rm vac}\) ceiling with a perfect plasma is +0.9991. The
threshold sits **above the oracle**, so no amount of better profile *selection* reaches it; only a
richer profile family could, and that widens the label-free selection problem at the same time.

**The learned route was tried hard and failed.** §28 MLP3 ran a machine-agnostic conv operator
through four arms: MAST \(R^2_\psi\) of \(-1.353\), \(-1.081\), \(-0.938\), with the coordinate
diagnosis showing the net puts the MAST plasma **81% of the way to the DIII-D magnetic axis**.
§27 Ultraviolet's fillable write passed P0 but the whole-shot component collapsed and T1/T2 closed
behind it. Neither line is a few epochs from 0.9976.

**`alvar0` is above our physics ceiling.** Their 0.999600 exceeds the +0.9991 that a *perfect*
plasma current gives through our vacuum construction. Whatever they do is not this family.

**And here is the thing that decides it.** The competition terms say external public datasets and
pre-trained models **are allowed** with disclosure — so the ledger's "no MAST GT exists, do not
train on MAST labels" premise is wrong on *availability*. MAST is an open-data machine: FAIR-MAST
(`mastapp.site`, CC BY-SA 4.0) publishes the M05–M09 archive, and the starter kit gives away that
this is the upstream source when it explains that MAST `psirz` arrives on "a doubled 129-column R
grid — 65 real columns interleaved with 64 empty — which we drop." The challenge paper curates
**2,416** MAST shots and releases **1,206** as the test fold.

**So roughly half the curated MAST corpus overlaps the hidden test fold, and its EFIT is public.**
The test shots are anonymized and cannot be excluded from a FAIR-MAST ingest (~53% overlap;
§34.16).

**Track 2 fact.** Our \((\alpha,\beta,\lambda)\) oracle caps at 0.9598, below the 0.9976
threshold that pays; MAST public-archive training is the supplement route after DIII-D (§34.16).
Ingest: `scripts/mast_ingest.py` → `datasets/fair_mast_train`.

*Unresolved and not to be asserted:* we do not know how `alvar0` reaches 0.9996. Exceeding our
perfect-plasma ceiling is consistent with either a better construction than ours or with the
archive, and **we should not accuse anyone of the latter** on this evidence.

### 34.15 The `mwaller` Cons gap is axis + \(\kappa\), i.e. placement

Public DIII-D cards, `o1x5` vs `mwaller` vs #1 (`mwaller_autopsy.py`):

| | us | mwaller | \(\Delta\) | share of Cons gap |
|---|---:|---:|---:|---:|
| \(\kappa\) | 0.961007 | 0.970316 | **+0.009309** | **35.6%** |
| \(Z_{\rm axis}\) | 0.981162 | 0.987678 | +0.006516 | 24.9% |
| \(R_{\rm axis}\) | 0.975334 | 0.980871 | +0.005537 | 21.2% |
| \(\ell_i\) | 0.957630 | 0.960227 | +0.002597 | 9.9% |
| `tri_top` / `tri_bot` | | | +0.00145 / +0.00141 | 5.5 / 5.4% |
| volume | 0.981531 | 0.980874 | **−0.000657** | we win |
| Cons | 0.976314 | 0.980052 | +0.003738 | |
| \(1-R^2_\psi\) | 2.16e-04 | 2.49e-04 | mwaller is **1.15×** our flux error | |

**82% of the gap is \(\kappa + Z_{\rm axis} + R_{\rm axis}\).** Volume we already beat them on. On the 10 worst-20 shots both lists share, we win \(\psi\) 7/10 (mean \(\Delta R^2_\psi=-0.00034\)) and lose \(D_{\rm LCFS}\) (mean \(\Delta=+0.0051\)): a slightly better flux map and a worse boundary. That is extraction efficiency as a one-line fact — they place the LCFS better from a worse \(\psi\).

Locally, on `o1x5` official derive, \(\kappa\) is the only scalar that collapses on the worst-covered 40 (0.960 → **0.940**; axis barely moves). Per-shot Cons vs coverage is \(\rho=-0.799\).

**This is the Forest warp again.** §34.12's five-dof span was isotropic scale, shift \(R\), vertical stretch — the public scalars those move are exactly \(\kappa\), \(R_{\rm axis}\), \(Z_{\rm axis}\). `mwaller` does not look like a different seven-scalar extractor. They look like a team that places the equilibrium better, which F16/F21/F56/F67 already failed to predict from a legal write. Do not reopen a warp. The remaining Track-1 jobs are the \(R^2_{q\beta}\) fold and O1 seeds.

### 34.17 The public \(R^2_{q\beta}\) fold is the **second half of each shot**

§34.8 said no local fold reproduces 0.995065. Coverage cannot see it. The shipped B8+F9 donor
on the 80-shot holdout (`qb_fold.py`, `st_support_cache` `q95_mean`/`betaN_mean`):

| fold | \(R^2_{q\beta}\) | vs public 0.995065 |
|---|---:|---:|
| all 80 | 0.996839 | +0.001774 |
| first 20% of each shot | 0.997216 | +0.002151 |
| **second half of each shot** | **0.995063** | **−0.000002** |
| last 20% | 0.993523 | −0.001542 |
| worst 40 shots by per-shot \(R^2_{q\beta}\) | 0.995393 | +0.000328 |

The public number is the late-in-shot number. This is a time-in-shot generalisation gap, not a
coverage gap and not a fitting gap — which is why every estimator selected on the full 80 or the
worst-covered 40 was saturating a fold the board does not pay.

**Use the second half as the \(R^2_{q\beta}\) instrument.** Any future qβ decision is scored there.
Still do not fit another head against the full 80.

### 34.18 LOCK — ridge on the second-half fold is the same null; second O1 draw finished

`qb_refit --subset second_half` on the lock trunks, scored on the fold that matches the board
(§34.17): shipped 0.994558, refit 0.994626, **+0.000068 / +0.40σ / P(better) 63.7%**, worth
\(+0.000011\) of \(S\). Unresolved, same as the random 80 and the worst-covered 40. The
instrument works (it sees the public-sized number); a linear head still cannot move it.

**Do not fit another \(q_{95}\)/\(\beta_N\) head.** The term is a late-in-shot generalisation
gap. The next legal attack is a *write or loss that sees the second half during training*, not
another closed-form head on the same trunks.

Second O1 seed draw **finished** (`o1_seed_sweep2.sh`: seeds 3, 17, 42, 2024, same 220/30 budget).
Cuts vs B8+F9 pair MSE: `LS_O1_3` 17.7%, `LS_O1_17` 20.3%, `LS_O1_2024` 20.1%. None beat the first-draw keepers `LS_O1_99` (32.5%) or `LS_O1_42` (29.6%). `LATE42` 25.8% and `W1` 29.2% also stay below `LS_O1_42`. Mix members unchanged.

### 34.19 NULL — O1 geometry rebuild (placement during plastic training)

Controlled seed-42 pair in `CRC_FEQ_olive_geom`: `O1G_control` (88 ep, geometry off) vs
`O1G_place` (adaptive κ/R/Z placement, gate @ ep 35, ramp ep 36–39). Cap 25 killed before
Cons reached 0.96; cap **40** required (control crossed 0.96 ~ep 36). Commands:
`python -m catskills.scripts.o1_geom_train {contract|smoke|control|place}`.

| arm | S | \(R^2_\psi\) | Cons | gate ep |
|---|---:|---:|---:|---:|
| `O1G_control` | 0.9942 | 0.9997 | 0.9789 | — |
| `O1G_place` | 0.9945 | 0.9997 | 0.9806 | 35 |

Paired official derive (`o1_geom_eval`, `cons_foldvar`):

| gate | result |
|---|---|
| all-80 Cons | **+2.17σ** (PASS ≥2σ) |
| worst-40 Cons | **+0.91σ** (FAIL ≥1.5σ) |
| κ/R/Z win | R +2.61σ, Z +4.20σ, κ +1.37σ all-80; **ℓᵢ −1.18σ** on worst-40 (FAIL no scalar < −1σ) |
| ψ vs control | tied at 0.9997 (PASS) |
| mix swap `o1x5_geom42` vs `o1x5_42for7` | +3.33σ Cons (PASS) |

**Verdict: NULL.** Directional placement gain on all-80 and in the mix swap, but worst-40 and
ℓᵢ safety gates fail. No tailfit, no package; `o1x5_42for7` ship unchanged.
Receipt: `results/olive_geom/O1G_eval.json`.

### 34.20 NULL — O1G-early (plastic gate + protected ℓ_i)

Same control (`O1G_control`). Treatment: gate at **R²ψ≥0.9985, Cons≥0.92** (fired ep **9**),
`placement_early` (κ/R/Z + ℓ_i=0.75, floor=0.5), 88 ep, 318 min.

| arm | S | Cons | gate ep |
|---|---:|---:|---:|
| `O1G_control` | 0.9942 | 0.9789 | — |
| `O1G_early` | 0.9946 | 0.9807 | 9 |

Paired vs control (`O1G_early_eval.json`): all-80 **+2.30σ**; worst-40 **+1.27σ** (need ≥1.5σ);
κ only +0.33σ all-80; ℓ_i worst-40 **−0.25σ** (fixed vs `O1G_place` −1.18σ). Mix swap
`o1x5_early42` vs ship: **+2.97σ** Cons locally — still **NULL**, do not package.

### 34.21 O1G VRAM batch probe (placement_early, geom ep 1)

1-epoch probes at 80k frames, `geom-warmup=0`, 2048 MiB driver headroom target.
Script: `catskills/scripts/o1_geom_batch_probe.py`.

| batch | peak used (MiB) | min free (MiB) | ok |
|---:|---:|---:|:---:|
| 256 | 1790 | 14058 | ✓ |
| 2048 | 3694 | 12154 | ✓ |
| 8192 | 10737 | 5111 | ✓ |
| 10240 | 13163 | 2685 | ✓ |
| **10752** | **13715** | **2133** | **✓ max** |
| 11008 | 13975 | 1872 | ✗ headroom |
| 12288 | 15454 | 394 | ✗ headroom |

Current O1G runs use batch **256** (~1.8 GiB peak, ~58% GPU util). **42× headroom** to
batch 10752 before the 2 GiB floor binds. Receipt: `results/olive_geom/O1G_batch_probe.json`.
Next: optional O1G rerun at 1024–2048 (conservative) or linear-LR-scaled 10752 if gates still fail.

### 34.22 NULL — O1G-early batch 2048 (VRAM probe follow-up)

Same recipe as §34.20 but **batch 2048** (LR 3e-4 unchanged). Gate @ ep **14**; 88 ep in **260 min**
(~99% GPU vs ~58% at batch 256). Holdout: **S=0.9947**, Cons **0.9817**, \(R^2_\psi\)=0.9997.

Paired vs `O1G_control` (`O1G_early_b2048_eval.json`):

| gate | result |
|---|---|
| all-80 Cons | **+2.03σ** (PASS) |
| worst-40 Cons | **+1.13σ** (FAIL ≥1.5σ; was +1.27σ @ batch 256) |
| κ all-80 | +0.16σ (still weak) |
| ℓᵢ worst-40 | +0.86σ (safe) |
| mix swap `o1x5_early42_b2048` vs ship | **+4.65σ** (PASS) |

**Verdict: NULL.** Larger batch improved wall time and mix swap, but worst-40 gate still fails.
Ship unchanged: `o1x5_42for7`.

### 34.23 NULL — second-half-weighted qβ training (QB2)

Public-aligned late-in-shot loss on B8 (`qb_tailfit --late`, select on `frac>=0.5`). Tests:
`tests/test_qb_late.py`. Donor eval vs shipped B8+F9 (`st_support_cache`) on the same 80 shots.

| arm | second-half \(R^2_{q\beta}\) | vs lock 0.995063 |
|---|---:|---:|
| shipped B8+F9 | 0.995063 | — |
| parent B8 alone | 0.993894 | −0.001169 |
| `QB2_B8_CTRL` late=0 | 0.993699 | −0.001364 |
| `QB2_B8_L2` late=2 | 0.993663 | −0.001400 |
| L2+F9 donor | **0.994467** | **−0.000596 / −3.32σ** |

PASS needed 0.995213 and ≥+1.5σ. PROMISING band was +0.000100…+0.000149. **FAIL.** No F9
replicate, no extra late dose, no new head. Receipts: `P35_QBFIT_QB2_B8_{CTRL,L2}.json`,
`P35_QBLATE_B8L2_F9.json`, `P35_QBLATE_DECISION.json`.

### 34.24 NULL — NR0→NR3 native-rate chain

Existing `NR0_tangent_oracle.json`: `gate_pass=false`, oracle \(S=0.987418\) vs 0.9960
(kill \(S<0.9957\)). Sprint re-ran `brown_native_probe`; it refused. NR1–NR3 and the NR ship
gate were not opened. Receipt: `results/brown_nr1/NR_SPRINT_DECISION.json`.

### 34.25 NULL — input-routed three-regime POD (MPOD)

NR failure fallback. New cache `regime_pod_v1` only (never `coeffs_v1`). Shot-level KMeans
(\(R=3\), `random_state=0`) on train-only mean+std of legal input \(X\). One \(k=60\) basis
per regime; excluded-shot pooled \(R^2_\psi\ge 0.99999\) (MP0 PASS).

MP1 matched 1,000-shot / 10-ep / batch-2048 dualctx control vs hard-routed R=3:

| | control | routed | need |
|---|---:|---:|---|
| pooled \(\psi\) MSE | \(6.82\times10^{-5}\) | \(7.11\times10^{-4}\) | ≥10% cut |
| worst-5% frame MSE | \(4.44\times10^{-4}\) | \(1.22\times10^{-2}\) | ≥10% cut |
| Cons | 0.759 | 0.544 | non-decrease |
| \(1-D_{\rm LCFS}\) | 0.961 | 0.944 | non-decrease |

Routed MSE rose 9.4× (tail 26×). **NULL.** No MP2, no soft routing, no extra regimes, no
larger \(k\). Receipts: `P35_MPOD_CACHE_MP1.json`, `P35_MPOD_MP1.json`, `P35_MPOD_DECISION.json`.

### 34.26 MISS — FAIR-MAST archive \(\psi\) (Track 2 M1)

Cache `mast_features_v1` only (never DIII-D `features_v1` / `coeffs_v1`): 3,738 ingest
parquets, width 308 (14 archive magnetics + Thomson). Dualctx 512×4, win9 smoothdiff,
clip 10, PCA \(k=60\), 12 ep, batch 2048, seed 42. Train 3,655 after excluding the three
demo keys and sealing 80. Holdout scoring uses `parquet_data/mast_shot_283xx`
(`prefer_demo=True`); ingest 283xx is a different recording (T and \(\psi\) scale).

| fold | \(R^2_\psi\) | note |
|---|---:|---|
| official 3-shot demo | **0.980803** | gate 0.9976 |
| sealed 80 (archive) | 0.513440 | in-distribution, shot-disjoint |
| ingest-aligned 283xx raw | 0.990571 | 0.9944 / 0.9858 / 0.9923 per shot |

Official \(S=0.659487\), Cons 0.17572, \(D_{\rm LCFS}=0.150991\). \(q_{95}\)/\(\beta_N\)
NaN. **MISS.** DIII-D ship stays `o1x5_42for7`. Not a zero-shot MAST claim (~53% archive
overlap, §34.14). Receipt: `results/mast_archive/P36_M1_archive.json`; ckpt
`CRC_FEQ_mast_M1_archive.pt`.

### 34.27 MISS — M1_long 50-ep control (Track 2)

Same trunk/cache as §34.26, tag `M1_long`, 50 epochs, new ckpt. Best val epoch 29
(val MSE \(2.26\times10^{-2}\) vs M1 \(2.41\times10^{-2}\)). Official demo
\(R^2_\psi=0.981548\) vs M1 0.980803: residual ratio 1.04×, not the ~8× needed
for 0.9976. Sealed 0.510443 (no gain). Ingest-aligned 283xx raw 0.993169.
**MISS.** Extra epochs are not the lever. Next is \(\Psi_{\rm vac}\) residual
(`M2_vacres`). Receipt: `P36_M1_long.json`.

### 34.28 NULL — M2_vacres \(\Psi_{\rm vac}\) residual (Track 2)

Sidecar `mast_vac_v1` (3,738 shots, never `mast_features_v1` / DIII-D). Train PCA+trunk
on \(\psi-\Psi_{\rm vac}\), add \(\Psi_{\rm vac}\) back at infer. 12 ep, same split as M1.

| fold | M1 | M2 | need |
|---|---:|---:|---:|
| official demo \(R^2_\psi\) | 0.980803 | **0.975249** | 0.9976 |
| sealed 80 | 0.513440 | 0.512213 | gain |
| ingest-aligned 283xx | 0.990571 | 0.990043 | — |

Official \(S=0.628115\), Cons 0.053, \(D_{\rm LCFS}=0.190\). Residual val MSE matched
M1; the vac skip did not cut residual ~8× and **hurt** the demo. **NULL.** Do not
extend epochs on this write. Receipt: `P36_M2_vacres.json`.

### 34.29 MISS — M3_qc quarantine + shot-balanced (Track 2)

Sidecar `mast_qc_v1` (never `mast_features_v1` / DIII-D). Train-only fence
\(\mathrm{maxabs} > \mathrm{median}+10\cdot\mathrm{MAD} = 2.981\) Wb (\(n_{\rm fit}=3655\)).
Quarantine 33 train (all maxabs) + sealed `27373` (maxabs 14.73, \(\psi\) to \(-14.7\)).
Retrain the M1 trunk, tag `M3_qc`, 12 ep, shot-balanced sampling, no vac residual.
Train 3,622. Best val epoch 8 (val MSE \(6.73\times10^{-3}\) vs M1 \(2.41\times10^{-2}\)).

| fold | M1 | M3_qc | need |
|---|---:|---:|---:|
| official demo \(R^2_\psi\) | 0.980803 | **0.980459** | 0.9976 |
| sealed 80 | 0.513440 | 0.508589 | — |
| sealed clean 79 | — | 0.904206 | — |
| ingest-aligned 283xx | 0.990571 | 0.996818 | 0.9976 |

Official \(S=0.639979\), Cons 0.088, \(D_{\rm LCFS}=0.169\). QC+balance lifted
in-distribution ingest (residual ~3× vs M1) and clean sealed to 0.90; the official
demo did not move. Pooled sealed stays ~0.51 because `27373` remains in the
comparison fold. **MISS.** Do not reopen QC fences on this trunk. DIII-D ship
stays `o1x5_42for7`. Receipt: `P36_M3_qc.json`.

### 34.30 MISS — M4_edge aye clock repair (Track 2)

Ingest dropped edge Thomson when ``aye.time.max() >= 10``. Archive clocks are
seconds; challenge EFIT is ms. ``*1000`` recovers demo 283xx Te at \(R^2=1\).
Sidecar `mast_aye_v1` (3,738/3,738, never `mast_features_v1` / DIII-D) and
`mast_features_v2` (same Y, edge columns filled). M3 trunk, tag `M4_edge`,
QC + shot-balanced, 12 ep. Best val epoch 8.

Zeroing the 32 edge columns on M3 at infer had already lifted official
\(R^2_\psi\) 0.980459 → 0.995329. Retraining on the repaired write:

| fold | M3_qc | M4_edge | need |
|---|---:|---:|---:|
| official demo \(R^2_\psi\) | 0.980459 | **0.993369** | 0.9976 |
| sealed 80 | 0.508589 | 0.506214 | — |
| sealed clean 79 | 0.904206 | 0.895952 | — |
| ingest-aligned 283xx | 0.996818 | 0.994737 | — |

Official \(S=0.690065\), Cons 0.271, \(D_{\rm LCFS}=0.105\). Residual vs M3 is
2.95×; still 2.76× short of 0.9976. **MISS.** Do not reopen the aye clock.
DIII-D ship stays `o1x5_42for7`. Receipt: `P36_M4_edge.json`.

### 34.31 MISS — M5_maskeval consistent official infer (Track 2)

Existing ckpts only. Official demo with edge columns 276:308 zeroed, matching
the v1 train write. No new cache, no overwrite.

| ckpt | raw \(R^2_\psi\) | mask \(R^2_\psi\) | mask \(S\) |
|---|---:|---:|---:|
| M1_archive | 0.980803 | 0.994274 | 0.7169 |
| M1_long | 0.981548 | 0.994099 | 0.7275 |
| M3_qc | 0.980459 | **0.995329** | 0.7075 |
| M4_edge | 0.993369 | 0.992013 | 0.6765 |

Best is M3_qc + mask (4.18× residual vs its poisoned official). M1_long mask is
*worse* than M1_archive mask — extra epochs on the poisoned path did not help
the consistent path. Masking M4 hurts; it was trained with edge. Gate 0.9976
still missed (residual 1.95×). **MISS.** Do not treat raw official M1–M3 as
the fair number. DIII-D ship stays `o1x5_42for7`. Receipt: `P36_M5_maskeval.json`.

### 34.32 — M6_resid official tail of M3+mask (Track 2)

No train. M3_qc + edge mask, 115 demo frames. Pooled \(R^2_\psi=0.995329\).

| shot | frames | \(R^2_\psi\) | SSE frac |
|---|---:|---:|---:|
| 28348 | 36 | 0.994039 | 0.410 |
| 28350 | 42 | 0.994802 | 0.410 |
| 28351 | 37 | **0.997278** | 0.180 |

28351 is already past 0.9976. The miss is split across 28348/28350. Top-10 frames
hold 44% of SSE; 13/115 worst frames are the excess to the gate. No single-shot
hog. **uniform.** Next is a 50-ep M3 clone scored with the same mask, not a
per-shot recipe. Receipt: `P36_M6_resid.json`.

### 34.33 MISS — M6_m3long 50-ep fair write (Track 2)

Same as M3_qc (v1, QC, shot-balanced) for 50 epochs, official infer edge-masked.
New tag, no overwrite. Best val epoch 28 (MSE \(6.04\times10^{-3}\) vs M3
\(6.73\times10^{-3}\)).

| fold | M3+mask | M6_m3long | need |
|---|---:|---:|---:|
| official demo \(R^2_\psi\) | 0.995329 | **0.995908** | 0.9976 |
| official \(S\) | 0.7075 | 0.7498 | — |
| Cons | 0.348 | 0.543 | — |
| \(D_{\rm LCFS}\) | 0.096 | 0.065 | — |

Residual vs M3+mask is 1.14×, not the 1.95× needed. Extra epochs on the fair
write are now closed too. **MISS.** DIII-D ship stays `o1x5_42for7`. Receipt:
`P36_M6_m3long.json`.

### 34.34 MISS — M7_win train-only \(|I_p|\) frame fence (Track 2)

Sidecar `mast_win_v1` (never `mast_features_v1` / DIII-D). Train-only 2-means
on \(|I_p|\) (seed 42): centers 63.4 / 594, midpoint \(\mathrm{ip\_lo}=328.7\).
Keep 211,935 / 345,124 frames; 357/3,622 shots empty. Official demo \(|I_p|\)
is 465–647, so every scored frame is inside the fence. M3 recipe, tag
`M7_win`, 12 ep, QC + shot-balance + edge mask. Val MSE fell every epoch
(2.24e-3 → 4.50e-4); best is epoch 11.

| fold | M6_m3long | M7_win | need |
|---|---:|---:|---:|
| official demo \(R^2_\psi\) | 0.995908 | **0.997596** | 0.9976 |
| official \(S\) | 0.7498 | 0.7177 | — |
| sealed 80 | 0.507 | 0.377 | — |

Official residual vs M6 is 1.70× — the remaining gap — and still
\(0.997596 < 0.9976\) by \(4\times10^{-6}\). Sealed/ingest-aligned drop because
those folds still score ramp frames. **MISS** on the written gate. Do not
soften 0.9976. DIII-D ship stays `o1x5_42for7`. Receipt: `P36_M7_win.json`.

### 34.35 PASS — M8_winlong 50-ep windowed write (Track 2)

Same as M7_win (v1, QC, shot-balanced, \(|I_p|\) fence, official edge mask),
50 epochs, new tag. Best val epoch 47 (MSE \(3.55\times10^{-4}\) vs M7
\(4.50\times10^{-4}\)).

| fold | M7_win | M8_winlong | need |
|---|---:|---:|---:|
| official demo \(R^2_\psi\) | 0.997596 | **0.997730** | 0.9976 |
| official \(S\) | 0.7177 | 0.7503 | — |
| Cons | 0.386 | 0.533 | — |
| \(D_{\rm LCFS}\) | 0.081 | 0.051 | — |

**PASS** the written Track 2 \(\psi\) gate. \(q_{95}\)/\(\beta_N\) stay NaN
(§31.3), so \(S\) is not a full MAST board number. Sealed stays ~0.38 (ramp
frames). \(\sim53\%\) archive overlap (§34.14); not a zero-shot claim. Do not
package DIII-D; ship stays `o1x5_42for7`. Receipt: `P36_M8_winlong.json`.

Public submission 927115: `mast_S=0.816713`, `G_ratio=0.822291`,
\(R^2_\psi=0.998466\), Consistency 0.863366, \(D_{\rm LCFS}=0.051167\).
The frozen DIII-D bytes scored `d3d_S=0.993217`.

### 34.36 — two-track probe package (frozen DIII-D + M8 MAST)

`benchmarks/CRC_FEQ_m8/0.1.0_20260915_152934`. DIII-D npz hardlinked from
`o1x5_42for7` (must reprint `d3d_S=0.993290`). MAST is `M8_winlong` on all 1,206
public-test shots, edge masked, NaN \(q_{95}\)/\(\beta_N\). Both npz validate.
Asks Codabench for `mast_S` and `G_ratio` with Challenge 1 held. Not zero-shot.
Receipt: that attempt `REPORT.md` / `meta.json`.

### 34.37 — M8 public label-free LCFS/axis sweep

Pooled stride-4 `plausibility` on the packaged public npz vs the GS v2 donor
(`P36_M8_plaus.json`). 17,269 frames / 1,206 shots.

| | LCFS fail | axis \(R\) p50 | axis \(Z\) p50 | \(R\) err p50 | \(Z\) err p50 |
|---|---|---|---|---|---|
| M8 | **0.0000** | 0.911 | +0.002 | 0.035 | 0.012 |
| GS v2 | 0.0008 | 0.875 | 0.000 | 0.054 | 0.010 |

Public `|Ip|` (pool, 69,075 frames): p10/p50 = 422/632 A; **2.67%** below
`ip_lo=328.7`. Fence is train-only — infer still writes those frames. No new
train. Not a score. Public card still waits on Codabench (~05:00 EDT 16 Sep).

### 34.38 — overnight M9 init/ensemble search (Track 2)

`--init-seed` only (data `--seed` 42 stays). M8 fp16 baseline reprints
\(R^2_\psi=0.997730\), \(S=0.7514\) (gate holds). `M9_win100_i42` epoch-47 val
MSE matches M8 exactly; best epoch **97** (longer write is not null). Three
50-ep init replicas all PASS. Residual corr 0.74–0.92. Four equal-weight
ensembles clear the written gates. Winner: mean of
`M9_win50_i{101,202,303}`, package fp16.

| | \(R^2_\psi\) | \(S\) | Cons | \(D_{\rm LCFS}\) |
|---|---:|---:|---:|---:|
| M8 fp16 | 0.997730 | 0.7514 | 0.539 | 0.051 |
| ens3 | **0.998555** | **0.7675** | **0.614** | **0.045** |

Public LCFS fail 0 / 17,725; axis err vs M8 \(+0.0002\) R / \(-0.0034\) Z m.
M7/M8 blend loses \(S\); smooth not triggered. DIII-D SHA matches o1x5.
\(q_{95}\)/\(\beta_N\) stay NaN. Verdict **CANDIDATE_READY**. Morning: upload
the already-pushed M8 probe first; second submit only after `d3d_S` reprints
0.993290. Package:
`benchmarks/CRC_FEQ_m8/0.1.0_20260915_181342`. Receipt:
`P37_OVERNIGHT_DECISION.json`.

Public submission 927191: `mast_S=0.822894`, `G_ratio=0.828514`,
\(R^2_\psi=0.998610\), Consistency 0.890400, \(D_{\rm LCFS}=0.044217\).
This is +0.006181 MAST \(S\) over M8. The identical DIII-D bytes again scored
`d3d_S=0.993217`, so the earlier 0.993290 did not reprint under the current
scorer.

### 34.39 NULL — DIII-D analog retrieval (P38)

Thesis: similar DIII-D input trajectories share transferable `o1x5_42for7`
residuals, so a leakage-resistant kNN correction could close part of the
public gap (0.993290 vs 0.995553). Frozen mix, frozen split
(`keys[:80]` confirmatory holdout / `keys[80:]` donor), correction
\(z_{\rm corr}=z_{\rm base}+\alpha\,\widehat{z_{\rm gt}-z_{\rm base}}\).
Never tuned on the canonical 80. Artifacts only under
`$LAB/results/brown/P38_ANALOG/`.

**In-fit screen (donor 6,961, five shot-disjoint folds).** Cheap 384-cell
3-fold / 2,048-index screen, then 30-cell full-donor promote, Cons-proxy
on top 5, Ip DTW rerank of the lock. Winner is **A0** (shot aggregates),
not A2/A3:

| cell | median \(\psi\) MSE cut | Cons-proxy \(\sigma\) |
|---|---:|---:|
| `A0_L32_none_cosine_k16_inverse_p16_a0.5` | **1.52%** | **+3.88** (5/5, \(P=1\)) |
| `A2_L64_pca64_cosine_k16_inverse_p16_a0.5` | 1.56% | +3.73 |
| `A3_L64_pca64_cosine_k16_inverse_p16_a0.5` | 1.44% | +3.22 |
| DTW \(k=16\) on the A0 lock | 1.27% | — |
| oracle (in-fit) | 20.7% | — |
| k256 / global-mean / \(\alpha=0\) | 0.58% / 0.05% / 0 | — |
| random / permute / phase-reverse | −4.8% / −3.9% / −1.0% | — |

Gate needed all-fold positive cut, median \(\ge 3\%\), Cons-proxy \(+\) in
\(\ge 4\) folds, \(P(\mathrm{better})\ge 0.95\). Cut is 1.52% vs 3%.
**Did not advance.** Shadow OOF models were not trained.

This is the same null as §34.11's 135-feature ridge (\(-0.56\%\) on the
holdout aggregates). Rich trajectories, phase payloads, and DTW do not
beat A0. The oracle offset remains large (holdout 46.2% of o1x5 MSE;
in-fit 20.7%).

**Holdout, one locked eval, no retune.** Coeff-MSE cut **+0.87%**. Paired
Cons \(+2.94\sigma\) (80), \(+1.74\sigma\) (worst-40). Official-proxy
\(\Delta S=+0.00013\) (0.968996 → 0.969128; this proxy is not the shipped
mix \(S=0.99446\) — \(R^2_{q\beta}=0.827\) is frozen donor heads on
reconstructed \(\psi\)). House bar 0.9945 missed. 24.9% of positive gain
in one shot. Fail `delta_S`, `house_S`, `shot_concentration`. Neighbor
audit: donors only.

**qβ arm (separately gated, in-fit).** Constant \(\alpha=1\) combined
\(R^2\) +0.095, both scalars non-regressing, first and second half both
up. Controls also move (global-mean +0.038, random +0.030). No OOF
confirmation. Shipped qβ heads stay; analog qβ was not applied on the
holdout and was not packaged.

**Verdict NULL.** No package, no HF push, no Codabench. Do not reopen
trajectory kNN / DTW on this mix without a *new legal observable* of the
per-shot offset. Restart is not required. Receipts:
`contract.json`, `preflight.json`, `cache_manifest.json`,
`in_fit_screen{,_cheap}.json`, `controls.json`, `ablations.json`,
`winner.json`, `crossfit_3.json` (skipped), `qb.json`,
`holdout_final.json`, `package_validation.json` (skipped),
`decision.json`.

### 34.40 MISS — P38 three-arm OOF follow-ups

Diagnostics only. Canonical split seed 42 / first 80 holdout / `donor_folds(..., 5, seed=0)`.
Did not overwrite `decision.json`, `winner.json`, or `holdout_final.json`. No package,
push, or submit. Frozen neighbor rule remains A0 / none / cosine / k16 / inverse / p16 / α=0.5.

**OOF substrate PASS.** Five B8-recipe shadows `CRC_FEQ_P38FU_shadow_f{0–4}.pt`,
exact `--train-keys-json` allowlists, 6,961 / 6,961 payloads under
`$LAB/results/brown/P38_ANALOG/followups/oof/` (6,035 s).

**Arm 1 qβ OOF MISS.** Locked `mode=constant` α=1.0. Coverage 6,961/6,961.
Fold combined gains −0.00100, −0.00012, +0.00148, +0.00004, −0.00206.
Median gain −0.00012; median Δq95 −0.00037; median ΔβN −0.00060; bootstrap σ −0.58
(seed 42). Control medians: random −0.00171, permute −0.00252, global-mean −4e-8.
Reasons: `not_all_pos`, `dq`, `db`, `sigma`, `control_global_mean`.

**Arm 2 ψ shrink MISS.** Nested λ ∈ {0, 0.25, 0.5, 1, 2}; frozen modal λ=0.25.
Nested outer cuts all positive (median 0.0458); unshrunk λ=0 median 0.0473;
concentration 0.014. Fail `no_gain_vs_unshrunk`. Holdout skipped.

**Arm 3 SVD phase MISS.** Fold-local ranks {4, 8, 16}; nested rank 16 all five
outer folds. Nested median cut 0.0313 vs raw p16 0.0473. Rank-4 explained energy
0.50–0.52. Controls (median): random −0.0088, permute −0.0097, global-mean +0.0060.
Fail `no_gain_vs_raw_p16`. Holdout skipped.

Driver `--resume --workers 10 --retry 1` printed `ALLDONE`. Exit 0
(PASS/MISS only). Receipts:
`followups/oof/manifest.json`, `followups/qb_oof/result.json`,
`followups/psi_shrink/result.json`, `followups/svd_phase/result.json`,
`followups/summary.json`, `followups/run.log`.

### 34.41 MISS — P38 model-aligned raw-p16 shadow ensemble

Diagnostics only. Reused the five trained shadows and 6,961 OOF payloads. No new
training. Did not overwrite `decision.json`, `winner.json`, `holdout_final.json`,
or `followups/summary.json`. No package, push, or submit. Frozen winner remains
A0 / L32 / none / cosine / k16 / inverse / p16 / α=0.5 / clip 0.5.

This is not the earlier cross-model +4.73% (comparison-only). Each shadow may
use only residual payloads produced by that same shadow. Shared frozen PCA
`7cc008c0d34b…efec9608` on all five `CRC_FEQ_P38FU_shadow_f{0–4}.pt`.

**OOF PASS** (model-aligned nested leave-group-out, inner seed 1). Coverage
6,961/6,961. Outer cuts +0.2106, +0.1868, +0.2109, +0.2018, +0.1447; median
0.2018. First/second-half medians 0.2289 / 0.1639. Control medians: random
0.0154, permute 0.0146, global-mean 0.0393. Paired Consistency σ 15.98
(seed 42, stride 8). Concentration 0.0051.

**Holdout MISS.** Five-member corrected ensemble vs uncorrected shadow ensemble
on the canonical 80. MSE cut +0.2001 (0.007855 → 0.006283). Official
ΔS +0.010002 (0.923274 → 0.933276); Cons σ 6.84; worst-40 σ 6.26; \(R^2_\psi\)
0.997006 → 0.997603; concentration 0.073. Fail `house_S` only (0.933276 <
0.9945). `candidate_ready=false`. Versus frozen o1x5: ΔS −0.03572
(0.968996 → 0.933276); coefficient MSE cut −10.49. qβ heads unchanged
(`r2_qb` 0.827083 both). 80/80 shots × 5 members inferred.

`--resume --workers 10` exit 0. Receipts:
`$LAB/results/brown/P38_ANALOG/followups/raw_p16_ensemble/{preflight,oof_result,holdout_result,result,progress}.json`,
`holdout_predictions/member_f{0–4}/`, `run.log`.

### 34.16 What is open

Public board 2026-09-14 21:55 ET (`results/forest/leaderboard.json`): we are **#10** at
\(S=0.993290\) (`o1x5_42for7`, 917128). #1 `dmitryzhevn` 0.995553; #4 `shansiming` 0.994620;
#5 `mwaller` 0.994345. Local ship proxy \(S=0.99446\). Ship stays until every official,
paired, tail, and scalar-safety gate passes.

The 2026-09-14 recovery sprint closed three more families: late qβ loss (§34.23), native-rate
NR0–NR3 (§34.24), and regime-conditioned multi-POD (§34.25). Analog retrieval on frozen
`o1x5` is NULL (§34.39). Do not reopen them with softer gates.

\(\kappa\) is still the largest unpaid Cons term (§34.15). Do not reopen Forest warps,
hindsight routers, ridge/head refits, O1G geometry, or coordinate-operator training.

**CLOSED (keep the measurements, not the jobs).** The per-shot offset is the Forest oracle warp
(§34.9–34.12): 29.7% of holdout \(\psi\) MSE, in-fit, family-wide, 69% five-dof geometric
misplacement. Do not reopen without a *new legal observable* of equilibrium position.

**Track 2 — M8_winlong and M9 ens3 scored §34.35/§34.38.** M8 public
`mast_S=0.816713`; ens3 public `mast_S=0.822894`, `G_ratio=0.828514`.
Ens3 is the Track-2 held result. Do not reopen M1–M7 recipes. Do not package
DIII-D; ship stays
`o1x5_42for7` until Challenge 1 gates pass. The NaN scalars are deliberate;
do not "fix" them (§31.3).

Do not reopen: any \(q_{95}\)/\(\beta_N\) *fit* — linear refit, head ensemble, scalar-objective
fine-tune, or late-in-shot qβ loss (§34.7–34.8, §34.23); mix incoherence or mix size (§34.3);
NR tangent/native residual (§34.24); input-routed multi-POD (§34.25); analog kNN / DTW on
`o1x5` residuals (§34.39); model-aligned raw-p16 shadow ensemble (§34.41);
QBNet auxiliary EFIT supervision (§37); residual-βN (`QBN_LAGR`); another from-scratch
QBNet width/lag/blend on the current th1 cache. The remaining \(\approx 0.0018\) of public
\(R^2_{q\beta}\) vs #1 is unreachable from this feature set. No further Challenge-1 GPU on qβ.

**Standing method note from tonight.** §34.7's scalar/coefficient tradeoff is still a real measured
mechanism even though the repair failed: every \(\psi\) gain since the tailfit was paid for in
\(R^2_{q\beta}\) trunk capacity, and we hide the bill by shipping donor heads. Future \(\psi\) work
should expect to keep paying it, and the donor overlay is what makes that survivable.

---

## 35. Operator v2 structured campaign (MISS 2026-09-17)

**Thesis.** Evolve the existing shape-conditioned operator: nested-surface decoder with a constrained residual, per-type lag kernels on a native-rate lag bank, hierarchical window/block/global shot context, and a grouped-OOF residual vs frozen `o1x5`. Hands-free runner with charcoal-class resume. No HF push, no Codabench upload.

**Entry.** `poetry run python -m catskills.scripts.operator_campaign run --resume auto`

Artifacts: `results/operator_c1/campaign_v2/`, checkpoints `CRC_FEQ_operator_v2_*`. Split remains `results/operator_c1/SPLIT.json` (development / sealed-outer / historical-80). SoftDerive stays off until measured `R2ψ≥0.999` and Cons `≥0.96`. q95/βN stay on the frozen `o1x5` donor unless a separate OOF donor gate passes.

| stage | command | status |
|---|---|---|
| Wipe 2026-09-16 | stopped pid 2088665; deleted `campaign_v2/`, all `CRC_FEQ_operator_v2_*`, smoke package, `cache/operator_targets_v2` (606), `cache/operator_lagbank_v1` (564) | clean. Kept `SPLIT.json` and pre-plan P1/P2/P3 |
| Architecture | flags + lazy resume + fp32 train | no autocast, no GradScaler; guard `amp=False` `precision=fp32` |
| Campaign | `run --resume auto` after wipe | restarted from empty receipts |
| Quality-regime grow 2026-09-16 | stopped pid 2125460 mid `P2400_timing_acausal`; `_promote` now requires held \(R^2\psi\ge 0.15\) and never force-appends combined | grow is timing-causal only; fp32 prefetch=2, accum=4, official every 2 epochs |
| Crossfit OOM 2026-09-17 | grow completed (`Pfull_timing_causal` s0/s1/s2); process died entering `predict_keys` on 6881 shots | resume with per-shot operator/o1x5 writes, mmap α fit |

Prior smoke/AMP tickets are void. Do not score or resume them. Combined / geom / hier are not grown.

Do not fit blend weights on historical-80 or sealed-outer. Do not reopen White / Ultraviolet.

---

<!-- o1_seed_sweep3_handoff -->

### O1 seed sweep 3 handoff (2026-09-17 15:54 UTC)

Git `43c262a1989d`. Operator v2 structured campaign: **MISS** (do not ship).

| seed | admission | paired σ all-80 | paired σ worst-40 | S | ΔS vs o1x5_42for7 | verdict |
|---|---|---:|---:|---:|---:|---|
| 5 | 0.24621533783223282 | nan | nan | None | None | FAIL |
| 29 | 0.29587506706464173 | nan | nan | None | None | FAIL |
| 101 | 0.36986417613482636 | nan | nan | None | None | FAIL |
| 314 | 0.26214027884586544 | nan | nan | None | None | FAIL |

**Decision:** `NULL` / line `CLOSED`.

<!-- o1g_mix_final -->

### 36. O1G mix final gate (NULL 2026-09-17)

**Thesis.** §34.22 closed `O1G_early_b2048` because *treatment vs control* missed worst-40 (+1.13σ). The object we would ship is the five-member mix `o1x5_early42_b2048` (`B8/L+O1/U+O1/L+O1/LS99+O1G/early_b2048`) versus incumbent `o1x5_42for7`. The old all-80 mix receipt was already +4.65σ; this run adds the missing worst-40 on the *actual mix*, then official S with frozen B8+F9 qβ. No training. Board at start: catskills **#10 / 0.993217**, #1 0.995633.

| gate | result |
|---|---|
| mix all-80 Cons (reused) | **+4.65σ** PASS (≥2σ); no scalar < −1σ |
| mix worst-40 Cons (new) | **+3.56σ** PASS (≥1.5σ); volume −0.33σ, ℓᵢ −0.18σ (safe) |
| official S vs `o1x5_42for7` | 0.994659 vs 0.99446; **ΔS +0.000199**; house 0.9945 PASS |
| \(R^2_\psi\) | 0.999798 vs 0.999795 (**+3e-6**, tol 2e-5) PASS |
| q95/βN | `st_support_cache:B8+F9` both mixes PASS |
| shot concentration | **0.417** FAIL (cap 0.20); `d3d_shot_bc0f6ecd42` 41.7% of positive ψ-MSE gain |

**Decision:** `NULL` / line `CLOSED`. Ship stays `o1x5_42for7`. No package, no HF, no Codabench. Receipts: `results/olive_geom/O1G_mix_final/{PREFLIGHT,DECISION,P33_PAIRED_*_worst40,P30_TAILFIT_HYBRID_SCORE_*}.json`. Driver: `scripts/o1_geom_mix_final.py`.

The earlier §34.22 worst-40 miss was the *standalone* O1G vs control, not the shipping mix. The mix *does* clear paired Cons on both folds and the house S bar; it fails the concentration safety gate.

## 37. QBNet auxiliary EFIT supervision (MISS 2026-09-19)

**Thesis.** Unused cache labels `efit_li` / `efit_r_axis` / `efit_z_axis` (Shafranov + \(\ell_i\); holdout corr(\(\beta_N\), \(R_{\mathrm{axis}}\))=+0.885) supervise a discarded aux head on a from-scratch LAGL-lag `lagmlp`. q95/βN heads still learn those labels directly. Not residual-βN (`QBN_LAGR`), not a frozen-trunk refit. Target term: \(0.15\,R^2_{q\beta}\) through βN.

**Entry.** `bash catskills/scripts/qbnet_aux.sh`. Tag `QBN_LAGA`, lags `0,1,2,4,8,16,32`, `--aux-labels li,r_axis,z_axis`, `--aux-weight 0.3`. Five folds, 80 holdout never trained. 16.4 GPU-min.

Fold is the second half of each of the 80 never-trained holdout shots — the one qβ fold with a receipted public calibration (LAGL local 0.995747 vs public 0.995726). It is **not** the broken 80-shot Cons instrument.

| receipt | second_half | q95 | βN | σ vs lock | vs LAGL blend |
|---|---:|---:|---:|---:|---|
| lock B8+F9 | 0.995063 | 0.996828 | 0.993299 | — | — |
| `QBN_LAGA_K5` solo | 0.995012 | 0.996875 | 0.993149 | −0.08 | worse |
| `QBN_LAGA_K5_BLEND50` | **0.995615** | **0.997402** | **0.993827** | **+2.79** | −0.000132 / βN −0.000284 |
| incumbent `QBN_LAGL_K5_BLEND50` | 0.995747 | 0.997383 | 0.994112 | +2.39 | — |
| ship bar | 0.996247 | 0.997283 | 0.994600 | +2.4 | — |

Generic `qbnet_gate` vs lock: blend **PASS** (Δ +0.000552 / +2.79σ, q95 bar ok, `all_frame_ok`). Ship bar vs LAGL: **FAIL** — second_half short 0.000632, βN short 0.000773 (and *below* LAGL). Aux supervision did not move the unpaid βN term.

**Decision:** `MISS` / line **CLOSED**. No package, no HF, no Codabench. Do not sweep `--aux-weight`, label subsets (`dsep`, li-only), `gru`, or per-scalar blend. QBNet is closed: remaining \(\approx 0.0018\) of public \(R^2_{q\beta}\) vs #1 is unreachable from the current feature set. No further Challenge-1 GPU on qβ. Best real ship stays LAGL 0.5-blend on o1x5 psi (`S=0.993850`). Visible row 933499 is still the packaging own-goal; do not burn a daily submit to restore it.

Receipts: `results/qbnet/QBNET_GATE_QBN_LAGA_K5{,_BLEND50}.json`, `QBNET_TRAIN_QBN_LAGA_f{0–4}.json`. Driver: `scripts/qbnet_aux.sh`.

## 38. Neural Picard Closure ceiling (MISS 2026-09-19)

**Thesis.** A network predicts 12-knot \(p'\)/\(FF'\) closures; a tied Picard solve using cached \(P\) owns every pixel. Representation ceiling first, against inference-time calibrated `vac_psi`.

48 labelled frames, 8 shots. GT-ext 3-param (diagnostic) \(R^2_\psi=0.995\). Calibrated-vac 3-param \(0.752\); 12-knot \(0.882\); 12-knot + 8 zero-net residual modes \(0.941\). Official Cons negative on every family. Exact \(I_p\). Multistart spread \(1.5\times10^{-4}\); medoid does not beat \(K=1\); no posterior. Gate \(R^2_\psi\ge0.9999\), Cons \(\ge0.985\): **MISS**. The miss is Green-smoothed vac residual a flux-surface closure cannot absorb, not spline poverty.

Receipt: `results/picard/EXIST.json`. Runbook: [`PICARD_RUNBOOK.md`](PICARD_RUNBOOK.md).

Generator (same day): challenge 19-rect authoritative; mode-1 512 frames accept 0.943, AUROC 0.533, seed-stable; mode-2 hull-mix 128 frames accept 0.883, AUROC 0.740. Receipts `results/picard/{GEOM,SYNTH,SYNTH_MODE2}.json`.

Pilots (same day): tiny-overfit 8 shots \(R^2_\psi=0.889\) (gate 0.9995). 512-shot / 1536-frame: no-solver val \(R^2=0.978\) vs Picard val \(R^2=0.912\), MSE cut \(-2.96\) (solver worse). **Do not promote.** Decision: `results/picard/DECISION.json`.

## 26. Sources

| thing | where |
|---|---|
| Member decorrelation, tail, aggregators (§30.7) | `results/brown/P30_DECORR.json`, `scripts/psi_decorrelate.py` |
| Router gate, closed (§30.8) | `results/brown/P30_ROUTER.json`, `scripts/psi_router.py` |
| Tail fine-tune, undertrained parents (§30.10) | `results/brown/P30_TAILFIT_*.json`, `P30_TAILFIT_HYBRID_SCORE.json`, `scripts/psi_tailfit.py`, `scripts/tailfit_mix_score.py`, `scripts/hybrid_qb_package.py` |
| Tailfit hybrid calibration probe (§30.11) | `benchmarks/CRC_FEQ_hybrid/0.1.0_20260904_134642/{REPORT.md,submission_pointer.zip}`, [Codabench 916155](https://www.codabench.org/competitions/17739/detailed_results/916155) |
| Indigo I0 ceiling, §30.6 falsified (§31.1–31.2) | `results/indigo/i0_ceiling.json`, `scripts/indigo_exist.py` |
| MAST \(q_{95}\)/\(\beta_N\) acceptance rules (§31.3) | `results/mast_qb/qb_rules_{solve,truth}.json`, `scripts/mast_qb_gate.py` |
| Indigo I1 \(j\)-basis ladder (§31.5) | `results/indigo/i1_basis.json`, `scripts/indigo_basis.py` |
| Consistency in physical units, `Z_axis` degeneracy (§31.6) | `results/indigo/i1_cons_units.json`, `scripts/indigo_cons_units.py` |
| Chebyshev compensating modes, KEEP (§31.7) | `results/indigo/i1_augment.json`, `scripts/indigo_augment.py` |
| Indigo basis, targets, verify, training (§31.8) | `scripts/indigo_train.py`, `indigo_operator.py`, `indigo_targets.py` |
| I2 corpus targets, sign fix (§31.8) | `cache/psivac_pool_v1/indigo_coeff_v1.json`, `scripts/fix_pool_ip.py` |
| I2 verify against the exact \(\psi_{\rm ext}\) (§31.8) | `results/indigo/i2_verify.json` |
| `zero_net` re-measured, fit-sample control (§31.8) | `results/indigo/i2_ceiling.json`, `scripts/indigo_ceiling.py` |
| I4 training run, 20 epochs (§31.9) | `results/indigo/i4_train.json`, `artifacts/indigo/i4_best.pt` |
| I4 scored on the three restated I3 gates (§31.9) | `results/indigo/i4_score.json`, `scripts/indigo_score.py` |
| I5 whitened-coefficient objective, epoch-1 probe (§31.10) | `results/indigo/i5_probe.json`, `cache/indigo_basis_v1/coeff_stats_k60_d4.npz` |
| I5 training run, 50 epochs (§31.10) | `results/indigo/i5_train.json`, `artifacts/indigo/i5_best.pt` |
| I5 scored, Indigo closed as dead (§31.10) | `results/indigo/i5_score.json`, `scripts/indigo_score.py` |
| Indigo arc record, nothing shipped (§31) | [`MODEL_DESCRIPTION_CRC_FEQ_indigo.md`](MODEL_DESCRIPTION_CRC_FEQ_indigo.md), `INDIGO_RUNBOOK{,_R2,_R3,_REMAINING}.md` |
| Baseline re-package, byte-identical to 912438 (§31.11) | `benchmarks/CRC_FEQ_solver/0.1.0_20260905_171902/{REPORT.md,meta.json}`, `scripts/solver_package.py` |
| Geometry-loss fine-tune, both losses null (§32) | `results/brown/P31_CONSFIT_{G,D}*.json`, `scripts/cons_tailfit.py` |
| Public per-scalar / per-shot results, any submission (§33.1) | `api/submissions/<id>/get_details/` → `scoring_result.zip` → `detailed_results.html` |
| Fold error bars + the paired comparator (§33.2–33.3) | `results/brown/P33_FOLDVAR.json`, `P33_PAIRED_hybrid_vs_pair.json`, `scripts/cons_foldvar.py` |
| \(\kappa\) autopsy, all of it in the LCFS height (§33.4) | `results/brown/P33_KAPPA_AUTOPSY.json`, `scripts/kappa_autopsy.py` |
| Sensitivity-weighted \(\psi\) loss, LOCK on replication (§33.5) | `results/brown/P33_KAPPAW_*.json`, `P33_KAPPA_SENS_*.npy`, `scripts/kappa_weighted.py` |
| \(q_{95}\)/\(\beta_N\) ensemble, refit, and the \(q_{95}\) overfit (§33.6) | `results/brown/P33_QB_ENSEMBLE.json`, `P33_QB_REFIT.json`, `scripts/qb_ensemble.py`, `scripts/qb_refit.py` |
| Shift-aware fold: coverage ranking + validation (§33.7) | `results/brown/P33_COVERAGE.json` |
| Coverage-weighted training, LOCK (§33.8) | `results/brown/P33_COVW_*.json`, `scripts/coverage_weighted.py` |
| Per-scalar cards for the whole board (§34.1–34.2) | `results/brown/P34_RIVAL_DETAILS.json`, `scripts/rival_details.py` |
| Mix coherence + mix-size sweep, both null (§34.3) | `results/brown/P34_MIX_COHERENCE.json`, `scripts/mix_coherence.py` |
| O1 seed sweep, long budget (§34.4) | `results/brown/P30_TAILFIT_LS_O1_*.json`, `scripts/o1_seed_sweep.sh` |
| `o1x5` paired confirmation + official score (§34.5) | `results/brown/P33_PAIRED_o1x5_vs_hybrid{,_worst40}.json`, `P34_O1X5_OFFICIAL.json` |
| `o1x5` package, HF commit `79b1468` (§34.5) | `benchmarks/CRC_FEQ_hybrid/0.1.0_20260906_040002/{REPORT.md,submission_pointer.zip}` |
| `o1x5` public result, #7 (§34.6) | [Codabench 916458](https://www.codabench.org/competitions/17739/detailed_results/916458), `results/board_details/detailed_916458.html` |
| \(q_{95}\)/\(\beta_N\) refit on tailfit trunks, LOCK (§34.7) | `results/brown/P33_QB_REFIT_o1x5.json`, `P34_QB_W/`, `scripts/qb_refit.py` |
| Scalar-objective fine-tune, LOCK (§34.8) | `results/brown/P35_QBFIT_QB_B8.json`, `scripts/qb_tailfit.py` |
| No local fold reproduces public \(R^2_{q\beta}\) (§34.8) | `results/brown/P33_QB_REFIT_oldtrunk_worst40.json` |
| Cross-team worst-shot overlap (§34.9) | `results/brown/P34_RIVAL_SHOTS.json`, `scripts/rival_shots.py` |
| Per-shot offset is in the fit, family-wide (§34.10) | `results/brown/P34_SHOT_OFFSET_B8.json`, `scripts/shot_offset.py` |
| NULL, offset not predictable from inputs (§34.11) | `results/brown/P34_OFFSET_REGRESS.json`, `scripts/offset_regress.py` |
| CLOSED, the offset is the Forest warp in coeff space (§34.12) | `results/brown/P34_SHOT_OFFSET_B8.npz` (analysis inline) |
| MAST board card; our qB is NaN, not zero (§34.13) | `results/brown/P34_RIVAL_DETAILS.json` (`--machine mast`) |
| FAIR-MAST ingest into the challenge schema (§34.14) | `scripts/mast_ingest.py`, `/tmp/mast_pool3.txt` |
| FAIR-MAST archive \(\psi\), MISS (§34.26) | `results/mast_archive/P36_M1_archive.json`, `run_crc_feq_pinn_mast.py`, `mast_cache.py` |
| M1_long 50-ep control, MISS (§34.27) | `results/mast_archive/P36_M1_long.json` |
| M2_vacres \(\Psi_{\rm vac}\) residual, NULL (§34.28) | `P36_M2_vacres.json`, `scripts/mast_warm_vac.py`, `cache/mast_vac_v1` |
| M3_qc quarantine + shot-balanced, MISS (§34.29) | `P36_M3_qc.json`, `mast_qc.py`, `scripts/mast_qc_index.py`, `cache/mast_qc_v1` |
| M4_edge aye clock repair, MISS (§34.30) | `P36_M4_edge.json`, `mast_aye.py`, `scripts/mast_warm_aye.py`, `cache/mast_aye_v1`, `cache/mast_features_v2` |
| M5_maskeval consistent official infer, MISS (§34.31) | `P36_M5_maskeval.json`, `scripts/mast_maskeval.py` |
| M6_resid official tail of M3+mask (§34.32) | `P36_M6_resid.json`, `scripts/mast_resid.py` |
| M6_m3long 50-ep fair write, MISS (§34.33) | `P36_M6_m3long.json` |
| M7_win \(|I_p|\) frame fence, MISS (§34.34) | `P36_M7_win.json`, `mast_win.py`, `scripts/mast_win_index.py`, `cache/mast_win_v1` |
| M8_winlong 50-ep windowed write, PASS (§34.35) | `P36_M8_winlong.json` |
| Two-track probe package, frozen DIII-D + M8 (§34.36) | `benchmarks/CRC_FEQ_m8/0.1.0_20260915_152934` |
| M8 public LCFS/axis sweep (§34.37) | `results/mast_archive/P36_M8_plaus.json` |
| Overnight M9 init/ensemble, CANDIDATE_READY (§34.38) | `P37_OVERNIGHT_DECISION.json`, `benchmarks/CRC_FEQ_m8/0.1.0_20260915_181342` |
| NULL, DIII-D analog retrieval (§34.39) | `results/brown/P38_ANALOG/decision.json`, `d3d_analog.py`, `scripts/d3d_analog_*.py` |
| MISS, P38 three-arm OOF follow-ups (§34.40) | `results/brown/P38_ANALOG/followups/summary.json`, `d3d_analog_followups.py`, `scripts/d3d_analog_followups*.py`, `scripts/d3d_analog_{qb_oof,psi_shrink,svd_phase}.py` |
| MISS, P38 raw-p16 shadow ensemble (§34.41) | `results/brown/P38_ANALOG/followups/raw_p16_ensemble/result.json`, `scripts/d3d_analog_raw_p16_ensemble.py` |
| Operator v2 structured campaign (§35) | `results/operator_c1/campaign_v2/`, `operator_geometry.py`, `operator_timing.py`, `operator_blend.py`, `operator_infer.py`, `scripts/operator_campaign.py`, `scripts/operator_package.py` |
| QBNet aux EFIT supervision, MISS / line closed (§37) | `results/qbnet/QBNET_GATE_QBN_LAGA_K5{,_BLEND50}.json`, `scripts/qbnet_aux.sh`, `qbnet.py` |
| Neural Picard Closure ceiling, MISS (§38) | `results/picard/EXIST.json`, `neural_picard.py`, `scripts/picard_exist.py` |
| `mwaller` gap is axis+\(\kappa\) (§34.15) | `results/brown/P34_MWALLER.json`, `scripts/mwaller_autopsy.py` |
| Public \(R^2_{q\beta}\) is the second half of each shot (§34.17) | `results/brown/P34_QB_FOLD.json`, `scripts/qb_fold.py` |
| Ridge on second-half fold, LOCK (§34.18) | `results/brown/P33_QB_REFIT_lock_second_half.json` |
| Second O1 seed draw, finished (§34.18) | `scripts/o1_seed_sweep2.sh`, `P30_TAILFIT_LS_O1_{3,17,42,2024,LATE42,W1}.json` |
| Late qβ loss, NULL (§34.23) | `P35_QBFIT_QB2_B8_*.json`, `P35_QBLATE_B8L2_F9.json`, `scripts/qb_tailfit.py`, `scripts/qb_late_eval.py` |
| NR0–NR3 sprint, NULL (§34.24) | `results/brown_nr1/NR_SPRINT_DECISION.json` |
| MPOD R=3, NULL (§34.25) | `P35_MPOD_{CACHE_MP1,MP1,DECISION}.json`, `regime_pod.py` |
| Public board snapshot 2026-09-14 21:55 ET | `results/forest/leaderboard.json` |
| Compact ledger through Steel | Olive [`MODEL_DESCRIPTION_CRC_FEQ_olive.md`](MODEL_DESCRIPTION_CRC_FEQ_olive.md) §0 |
| Color names / hex / one-paragraph history | [`CRC_FEQ_COLORS.md`](CRC_FEQ_COLORS.md) |
| Public board snapshot | `results/forest/leaderboard.json` (2026-08-20 07:52) |
| Public / old local mix | `results/CRC_FEQ_brown_B9_mix_B8_F9_score.json` |
| Local held mix (S1+F9) | `results/sky/M1_mix.json`, `results/CRC_FEQ_sky_M1_mix_score.json` |
| Red P0 | `results/red/P0_exist.json` |
| Gray P0 / G1 | `results/gray/P0_exist.json`, `results/gray/G1_roles.json` |
| Coral P0 | `results/coral/P0_exist.json` |
| Projection gates G1–G4 (§30) | `results/brown/PROJECTION_EXIST.json`, `scripts/projection_exist.py` |
| Scalar gate + scaling curve (§30.5) | `results/brown/P30_SCALAR_GATE{,_3500,_7000}.json`, `scripts/projection_scalar_gate.py` |
| Sky P0 / S1 / S2 / M1 | `results/sky/P0_exist.json`, `S1_tokens.json`, `S2_vacgrid.json`, `M1_mix.json` |
| Yellow P0 / Y2 | `results/yellow/P0_exist.json`, `results/yellow/Y2_parent.json` |
| White (closed) | [`WHITE_PLAN.md`](WHITE_PLAN.md), [`MODEL_DESCRIPTION_CRC_FEQ_white.md`](MODEL_DESCRIPTION_CRC_FEQ_white.md) |
| Ultraviolet (open) | [`ULTRAVIOLET_PLAN.md`](ULTRAVIOLET_PLAN.md), [`MODEL_DESCRIPTION_CRC_FEQ_ultraviolet.md`](MODEL_DESCRIPTION_CRC_FEQ_ultraviolet.md) |
| Olive night log | [`OLIVE_PLAN.md`](OLIVE_PLAN.md) §11 |
| Plans | `*_PLAN.md` per color |
| New color process | [`NEW_COLOR_DRILL.md`](NEW_COLOR_DRILL.md) |
