#!/usr/bin/env python3
"""CRC_FEQ_maroon phase chains — one run at a time on one GPU.

    poetry run python -m catskills.scripts.maroon_ladder p1 [wait_minutes]

Waves: ``m1`` is the completed data ladder, kept verbatim as the record of what it measured. ``p1``
onwards are the plan M1 replaced it with — see MODEL_DESCRIPTION_CRC_FEQ_maroon.md §10, and the
``PLATFORM`` comment below for the three timing runs that set their batch size and epoch budget.

Follows the charcoal chain's log protocol exactly, because ``maroon_watch`` parses it: ``### <tag>``
opens a run and ``### <tag> exit=<rc> <n> min`` closes it, with the runner's epoch lines between.

**M1 is the decision gate for the whole model.** Section 24 measured holdout R2_psi falling
monotonically as parameters grew at a fixed 200k frames, which at fixed data is overfitting and so
points at a data limit rather than a capacity ceiling. M1 tests that directly by holding *frame
presentations* constant and varying only how many unique frames those presentations come from:

    M1a_control_200k     901 shots    200,000 frames  x 420 epochs  = 84.0M
    M1b_mid_600k       2,721 shots    600,000 frames  x 140 epochs  = 84.0M
    M1c_full_1541k     6,961 shots  1,540,767 frames  x  55 epochs  = 84.7M

Three properties make this a controlled comparison rather than three runs:

* **Nested data.** All arms draw from one shuffled key list with one seed, so the control's 901 shots
  are a subset of the mid arm's 2,721, which are a subset of the full arm's 6,961.
* **One holdout.** ``sample_cache_keys`` shuffles the whole index before slicing, so all three arms
  score on the same 80 shots regardless of how many they train on.
* **Frozen basis.** The k=60 target basis comes from the M0 cache, identical for every arm. Charcoal
  re-fit PCA per run, which would have confounded "more data" with "different target axes".

Everything else is P8s's ``CAPS`` verbatim — geometry off, seed 42, val-fraction 0, patience 0, same
optimizer, same L-BFGS polish — so the control is comparable to P8f, which scored holdout R2_psi
0.99278. The one unavoidable deviation is the frozen basis, so the control is a re-run of P8f and not
a reuse of its number.

**Read:** holdout ``r2_psi`` per arm. If the full arm does not beat the control at equal compute, the
maroon thesis is wrong and M2 through M5 do not get spent.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from catskills import lab_paths as lp  # noqa: E402
from catskills.lab_paths import results_dir  # noqa: E402

MODEL_ID = lp.MODEL_ID  # imported, not re-declared: a color that exports CRC_FEQ_MODEL_ID must not
# end up with a different prefix here than in the runner it spawns.
MAROON_DIR = results_dir() / "maroon"
# Kept separate from MAROON_DIR because successor colors reuse this chain by redirecting the results
# directory (see purple_ladder), and they inherit maroon's *verified* frozen basis rather than re-gating
# it — so the M0 receipt must keep pointing at the run that actually measured it.
M0_GATE = MAROON_DIR / "maroon_m0_cache.json"

# P8s's CAPS, plus --coeff-targets. --max-train-shots is raised to the whole index so that
# --max-frames is the only thing that decides an arm's data; at 200k it trims to 901 shots, which is
# the same ~900 charcoal was actually getting.
BASE = [
    "--from-feature-cache", "--coeff-targets",
    "--max-train-shots", "7000", "--eval-shots", "80", "--seed", "42",
    "--batch-size", "64", "--n-pca", "60",
    "--optimizer", "adamw", "--weight-decay", "1e-4", "--lr", "1e-3",
    "--lbfgs-steps", "25", "--lbfgs-max-frames", "8192",
    "--temporal-window", "0", "--lambda-h1", "0", "--lambda-disp", "0",
    "--geom-emphasis", "0",
    "--val-fraction", "0", "--early-stop-patience", "0",
    "--hidden", "256,128", "--activation", "tanh",
    "--ckpt-every", "100",
]

# --------------------------------------------------------------------------- the revised plan (P*)
#
# Measured 2026-08-12 before P1 was launched, because the plan's 6 h estimate for it turned out to be
# wrong by 3x and the error was in one number nobody had measured: what the geometry loss costs at
# full-corpus scale. Three calibration runs, all at --max-frames 0 with --geom-emphasis 10:
#
#     batch 64,  geom_frac 1.00   464.3 s/epoch    GPU 47%
#     batch 64,  geom_frac 0.25   446.0 s/epoch    GPU 47%   (only 4% cheaper!)
#     batch 256, geom_frac 1.00   202.1 s/epoch    GPU 94%
#
# Two things follow. First, --geom-frac is NOT a speed lever here: quartering the frames the geometry
# term sees per batch bought 4%, because the cost is per-batch overhead in ``soft_derive`` (~17 ms x
# 21,665 batches), not per-frame work. Second, the lever that *does* work is the batch, for the same
# reason -- 4x fewer batches is 2.3x less wall clock for identical presentations, and it takes the card
# from half-idle to saturated.
#
# So the P phases run at batch 256. Note what that does to the step count: 250 epochs x 5,416 batches
# = 1.35M optimizer steps, which is within 3% of the 1.31M steps the shipped 420-epoch run took at
# 200k frames. P1 therefore matches the shipped run's optimization budget while seeing 4x the frame
# presentations -- and that, not epoch count, is what "converged" has to mean here. Geometry targets
# cost a further 87 s once, and all seven scalars are plausible on 100.0% of the corpus.
PLATFORM = [
    "--from-feature-cache", "--coeff-targets",
    "--max-train-shots", "7000", "--eval-shots", "80", "--seed", "42",
    "--batch-size", "256", "--n-pca", "60",
    "--optimizer", "adamw", "--weight-decay", "1e-4", "--lr", "1e-3",
    "--lbfgs-steps", "25", "--lbfgs-max-frames", "8192",
    "--temporal-window", "0", "--lambda-h1", "0", "--lambda-disp", "0",
    # val 0.1 is the shipped model's own setting (the split is always by shot), and it is what turns
    # on best-epoch selection against the S proxy. M1 ran val-fraction 0 and so could only take its
    # final epoch -- fine for a data ladder, wrong for a run we intend to build on. The shipped run
    # selected epoch 352 of 420, so the selector does move.
    "--val-fraction", "0.1", "--early-stop-patience", "0",
    "--hidden", "256,128", "--activation", "tanh",
    "--max-frames", "0",
    "--ckpt-every", "10",
]

# R1's config, so the R2 arms differ from it in exactly one flag each.
R1_TRUNK = ["--arch", "resnet", "--activation", "relu", "--trunk-width", "512",
            "--trunk-blocks", "4", "--trunk-dropout", "0.05", "--geom-emphasis", "0",
            "--early-stop-patience", "20", "--epochs", "80", "--log-every", "2"]

# Everything R7 established and R9 confirmed, minus the two axes R8 varies (width/blocks/epochs).
# The warmup and clip are not optional: they are the entire difference between R5 diverging at 17M
# and R7 converging at the same size. ckpt-every 5 makes each member resumable, so a wedge costs ~9
# min instead of the member.
STABLE = ["--arch", "resnet", "--activation", "relu", "--trunk-dropout", "0.05",
          "--geom-emphasis", "0", "--feature-clip", "10", "--early-stop-patience", "25",
          "--log-every", "4", "--lr-warmup", "5", "--grad-clip", "1.0", "--ckpt-every", "5"]

WAVES: dict[str, list[tuple[str, list[str]]]] = {
    # R2 -- the two cheapest levers on R2_psi, run as single-variable arms against R1 (S 0.97506,
    # R2_psi 0.99880). Everything else is R1 exactly: same trunk, same corpus, same frozen basis,
    # same 80-shot holdout, geometry off, 80 epochs at batch 256.
    #
    # Why psi and not Consistency: Consistency is 87% of the weighted gap to rank 1, but P1 measured
    # that training for it directly costs more than it buys, and R1 measured that it follows psi for
    # free (+0.183 with the geometry term off). The leaderboard says the same thing -- at R2_psi
    # ~0.9988 everyone lands at Consistency 0.87-0.90, and at >=0.9994 they land at 0.95-0.97.
    #
    # R2a, temporal context. --temporal-window 5 rebuilds each frame as [raw | 5-frame moving
    # average | central difference], 149 features -> 447. Charcoal shipped with this on; the maroon
    # data ladder turned it off to keep M1 a clean data experiment and it was never turned back on.
    # Rank 1's method name is "PCA50 + dual-CONTEXT ResNet flux ensemble", which is what this is.
    #
    # R2b, input hygiene. R1's psi_diag: the worst 5% of holdout shots carry 35.9% of the flux
    # residual, and they are the shots whose inputs reach |z|=53 on several columns; drop them and
    # mean R2_psi is already 0.99849. --feature-clip 10 clamps standardized inputs at 10 sigma,
    # which touches only the tail and leaves every in-range frame identical.
    #
    # GATE for both: holdout R2_psi > 0.99880, R1's own. Whichever arms clear it get combined.
    "r2": [
        ("R2a_temporal_w5", [*PLATFORM, *R1_TRUNK, "--temporal-window", "5"]),
        ("R2b_clip10", [*PLATFORM, *R1_TRUNK, "--feature-clip", "10"]),
    ],
    # --------------------------------------------------------------- the overnight chain (R4, R3, R5)
    #
    # Ordered cheapest-first so a crash at 03:00 still leaves the morning with results. R4 is 45 min,
    # R3 is 2.5 h, R5 takes the rest. Budgeted against a measured 27.8 s/epoch at width 512 and a
    # measured 71.3 s/epoch at width 1024 with 8 blocks -- 2.6x, not the 8x the parameter count
    # suggests, because these epochs are not purely trunk-bound.
    #
    # R4 -- the arm R2 never ran. R2a (temporal context) failed alone and R2b (10-sigma clip) passed
    # alone, and they are not independent: the central-difference block R2a adds is a derivative, so
    # it manufactures precisely the outliers R2b clips. Crossing them is the only way to know whether
    # temporal context was ever the problem.
    "r4": [
        ("R4_temporal_clip", [*PLATFORM, *R1_TRUNK, "--temporal-window", "5",
                              "--feature-clip", "10"]),
    ],
    # R3 -- the ensemble. Three more R2b models that differ ONLY in weight init and batch order
    # (--init-seed), never in --seed, so all four members share one holdout and the mean is scored on
    # the same 80 shots as every other number on the board. R2b itself is the fourth member.
    #
    # This is the highest-value arm and the reasoning is worth keeping: our three data points lie on
    # a line of ~60 Consistency per unit R2_psi (M1c 0.99573/0.7103, R1 0.99880/0.8936, R2b
    # 0.99895/0.9028), and on that line Consistency 0.97 needs R2_psi ~ 1.0000. Rank 1 has 0.9736 at
    # public R2_psi 0.9994, a slope near 23. Both cannot be the same mechanism, so Consistency
    # depends on the character of the flux residual and not only its size -- and averaging inits
    # changes character. A 2-member trial on 8 shots moved Consistency +0.041 for ψ +0.0003.
    "r3": [
        ("R3a_seed101", [*PLATFORM, *R1_TRUNK, "--feature-clip", "10", "--init-seed", "101"]),
        ("R3b_seed202", [*PLATFORM, *R1_TRUNK, "--feature-clip", "10", "--init-seed", "202"]),
        ("R3c_seed303", [*PLATFORM, *R1_TRUNK, "--feature-clip", "10", "--init-seed", "303"]),
        ("@ensemble:R3_ensemble4",
         ["R2b_clip10", "R3a_seed101", "R3b_seed202", "R3c_seed303"]),
    ],
    # R5 -- capacity and schedule, the last untested scale lever. 17.0M parameters against R2b's
    # 2.2M, on a cosine sized for 200 epochs rather than 80. R1's curve was still creeping up when
    # its 80-epoch schedule ran out, and 200 x 71.3 s is 4.0 h, which is what the night has left
    # after R4 and R3.
    "r5": [
        ("R5_wide1024_deep8", [*PLATFORM, "--arch", "resnet", "--activation", "relu",
                               "--trunk-width", "1024", "--trunk-blocks", "8",
                               "--trunk-dropout", "0.05", "--geom-emphasis", "0",
                               "--feature-clip", "10", "--early-stop-patience", "25",
                               "--epochs", "200", "--log-every", "5"]),
    ],
    # R7 -- R5 again, with the two things it lacked and nothing else. R5 is void rather than negative:
    # it diverged (loss 0.089 at epoch 5, 112 at epoch 10, 7.9e6 at epoch 20), early-stopped at 30 and
    # scored its epoch-5 weights, so 17M parameters were never actually tested. A 1024x8 residual
    # trunk starts at full lr with random skip paths and blows up in the first few epochs, so this
    # ramps the lr in over 5 epochs and clips the gradient norm at 1.0. Every other flag is R5's, byte
    # for byte, so a pass here is attributable to the stabilizers and a fail is attributable to
    # capacity. Epochs cut 200 -> 100 because R5 measured 65-70 s/epoch: ~2 h, not 4.
    #
    # Why capacity is worth one more try after R6's oracle ceiling: the basis supports Consistency
    # 0.9972 and we sit at 0.9231, so the entire gap is fit quality -- and a better base model lifts
    # every future ensemble member at once, which no amount of reseeding can do.
    "r7": [
        ("R7_wide1024_stable", [*PLATFORM, "--arch", "resnet", "--activation", "relu",
                                "--trunk-width", "1024", "--trunk-blocks", "8",
                                "--trunk-dropout", "0.05", "--geom-emphasis", "0",
                                "--feature-clip", "10", "--early-stop-patience", "25",
                                "--epochs", "100", "--log-every", "5",
                                "--lr-warmup", "5", "--grad-clip", "1.0"]),
    ],
    # R9 -- one more push on the BASE before eight members are committed, because base quality beats
    # member count outright: R8_curve prices 4 -> infinity members at +0.0023, while R2b -> R7 bought
    # +0.0051 in a single model and shifts the whole 1/N curve up with it.
    #
    # Framed as EQUAL WALL CLOCK, not equal epochs. Eight members is what we are buying, so the only
    # question that matters is what a config is worth per hour: R7 spent 2.0 h as 100 epochs at width
    # 1024, so this spends the same 2.0 h as 64 epochs at width 1280 (26M params, ~1.56x the cost per
    # epoch). A win means the eight members should be wider and shorter; a loss means R7's shape is
    # already the right way to spend an hour and the ensemble proceeds on it unchanged.
    #
    # Not tested here: whether a LONGER schedule at width 1024 beats both. R7's own tail cannot settle
    # that -- its val_psi decelerated over epochs 85-100 because a 100-epoch cosine drives lr to zero,
    # which is a property of the schedule and not evidence of convergence. That is the next probe if
    # this one loses.
    "r9": [
        ("R9_wide1280_64ep", [*PLATFORM, "--arch", "resnet", "--activation", "relu",
                              "--trunk-width", "1280", "--trunk-blocks", "8",
                              "--trunk-dropout", "0.05", "--geom-emphasis", "0",
                              "--feature-clip", "10", "--early-stop-patience", "25",
                              "--epochs", "64", "--log-every", "4",
                              "--lr-warmup", "5", "--grad-clip", "1.0",
                              # 5-epoch insurance: ~9 min of exposure now that partials resume.
                              "--ckpt-every", "5"]),
    ],
    # R8 -- the heterogeneous ensemble, and the reason it is not simply "eight more seeds".
    #
    # R8_curve scored all 15 subsets of R3's four members and both curves fit f(N) = f_inf - a/N at
    # R2 0.9999: S(N) = 0.98395 - 0.00910/N. So N=8 is worth +0.0011 over N=4 and N=inf is worth
    # +0.0023, against a gap of 0.0069 in projected public units. Averaging only cancels the error
    # that DIFFERS between members; what survives sets f_inf, and reseeding one architecture on one
    # corpus produces members whose systematic error is nearly identical. Hence: members that are
    # wrong in different ways, one axis moved at a time so a win stays attributable.
    #
    # DIVERSITY AXES, and why batch size is one of them. Six of these move capacity or input
    # conditioning (clip / width / depth / dropout), which change the function class. The seventh
    # moves the OPTIMIZER's noise: batch size sets the variance of each gradient estimate and so
    # which minimum a member falls into, which decorrelates error at least as strongly as a width
    # tweak and arguably more fundamentally, since two nets of different width fitted by the same
    # noisy path can still land in the same basin. It is also free here -- measured 2.9 GB of 12 GB
    # VRAM in use at 97% GPU utilisation, so the memory for batch 1024 is already idle, and being
    # compute-bound rather than memory-bound means the epoch cost barely moves. lr is sqrt-scaled
    # with the batch (1e-3 at 256 -> 2e-3 at 1024) because holding lr fixed while quadrupling the
    # batch shrinks the per-sample step and would confound "different noise" with "less training".
    #
    # Ordered cheapest-first: the narrow/shallow arms land early, so a stall still leaves members.
    # The four R3 members join the mean for free at the end -- they are already trained and they
    # share the data seed, so the final ensemble is 12 rather than 8.
    # R10 -- the two flux levers, because R8 projects to rank 2 and not rank 1.
    #
    # R8's ten-member mean projects to local S 0.98760 = PUBLIC 0.99080 against rank 1's 0.99186. It
    # does not fail, it MOVES the problem: Consistency falls from 86% of the gap to 11% and what is
    # left is R2_psi (93%) and D_LCFS (75%), offset by our q95/betaN surplus. Four members cut local
    # psi error only 1.33x, so count cannot finish it. Both arms use R7's stable trunk exactly, at
    # R7's 88-epoch budget, so each is a single-variable test against a measured control.
    "r10": [
        # (a) The context arm rank 1's method name points at, with the bug fixed. R2a and R4 fed the
        # trunk a ONE-frame difference of the RAW signal -- max|z| 105 with 24 of 149 columns above
        # 50 -- and both lost to no context at all. Differencing the SMOOTHED signal instead measures
        # max|z| 65 with 14 columns above 50 on the same 200 shots. Clip stays at 10 so this is one
        # variable against R4, not two.
        # PLATFORM sets --temporal-window 0; argparse takes the last occurrence, so the 5 below wins.
        ("R10a_ctx_smoothdiff", [*PLATFORM, *STABLE, "--trunk-width", "1024",
                                 "--trunk-blocks", "8", "--epochs", "88",
                                 "--temporal-window", "5", "--temporal-mode", "smoothdiff"]),
        # (b) The boundary term, never once optimized in the whole R-series. It carries 75% of the
        # post-R8 gap at weight 0.10. The risk is the trade that cost teal 0.0034 S -- buying boundary
        # accuracy by degrading the flux map -- so the gate demands D_LCFS up with psi held.
        ("R10b_lcfs", [*PLATFORM, *STABLE, "--trunk-width", "1024", "--trunk-blocks", "8",
                       "--epochs", "88", "--lambda-lcfs", "0.05", "--lcfs-emphasis", "10"]),
    ],
    # Re-centered after R7/R9. The old version of this wave hung eight members off R1's 2.2M trunk,
    # and the 1/N law rules that out arithmetically: R2b-class members reach 0.98286 at eight and
    # 0.98399 at infinity, while tying rank 1 needs ~0.9886 local. Members must come from the
    # stable-trunk class instead, which costs ~1.8 h each rather than ~35 min.
    #
    # Two shapes, not one, because R7 and R9 tied on S (0.97998 vs 0.98002) while erring DIFFERENTLY:
    # R7 is better on psi by 0.00018, R9 better on kappa by 0.0120. Equal quality with decorrelated
    # error is exactly what an ensemble wants, so both shapes are in the mix rather than only the
    # nominal winner. Capacity itself is spent -- 2.2M -> 17M bought +0.0030, 17M -> 26.5M bought
    # +0.00004 -- so these axes vary the ERROR, not the scale.
    #
    # Every member is budgeted to ~1.8 h using each shape's measured s/epoch (1024w 73 s, 1280w 95 s,
    # 768w 41 s, 12 blocks 110 s), so no member gets a quality edge from simply being given longer.
    # init_seed varies throughout while --seed stays 42, so all members share one holdout and the mean
    # is scoreable; R7 and R9 then join for free at the end.
    "r8": [
        ("R8a_w1024", [*PLATFORM, *STABLE, "--trunk-width", "1024", "--trunk-blocks", "8",
                       "--epochs", "88", "--init-seed", "811"]),
        ("R8b_w1280", [*PLATFORM, *STABLE, "--trunk-width", "1280", "--trunk-blocks", "8",
                       "--epochs", "68", "--init-seed", "822"]),
        ("R8c_w768_long", [*PLATFORM, *STABLE, "--trunk-width", "768", "--trunk-blocks", "8",
                           "--epochs", "158", "--init-seed", "833"]),
        ("R8d_deep12", [*PLATFORM, *STABLE, "--trunk-width", "1024", "--trunk-blocks", "12",
                        "--epochs", "59", "--init-seed", "844"]),
        ("R8e_clip5", [*PLATFORM, *STABLE, "--trunk-width", "1024", "--trunk-blocks", "8",
                       "--epochs", "88", "--init-seed", "855", "--feature-clip", "5"]),
        ("R8f_clip20", [*PLATFORM, *STABLE, "--trunk-width", "1024", "--trunk-blocks", "8",
                        "--epochs", "88", "--init-seed", "866", "--feature-clip", "20"]),
        ("R8g_drop015", [*PLATFORM, *STABLE, "--trunk-width", "1024", "--trunk-blocks", "8",
                         "--epochs", "88", "--init-seed", "877", "--trunk-dropout", "0.15"]),
        # Batch is the one axis that changes the OPTIMIZER's noise rather than the function class --
        # it sets which minimum a member falls into -- and it is free: 2.8 of 12 GB VRAM in use at
        # 99% GPU means the card is compute-bound and the memory for a 4x batch is already idle.
        ("R8h_batch1024", [*PLATFORM, *STABLE, "--trunk-width", "1024", "--trunk-blocks", "8",
                           "--epochs", "100", "--init-seed", "888",
                           "--batch-size", "1024", "--lr", "2e-3"]),
        # Heterogeneous-only, so the diversity claim is not carried by anything else.
        ("@ensemble:R8_ensemble8",
         ["R8a_w1024", "R8b_w1280", "R8c_w768_long", "R8d_deep12", "R8e_clip5", "R8f_clip20",
          "R8g_drop015", "R8h_batch1024"]),
        # R7 and R9 are the same class and already trained, so they cost nothing to add.
        ("@ensemble:R8_ensemble10",
         ["R8a_w1024", "R8b_w1280", "R8c_w768_long", "R8d_deep12", "R8e_clip5", "R8f_clip20",
          "R8g_drop015", "R8h_batch1024", "R7_wide1024_stable", "R9_wide1280_64ep"]),
    ],
    # R1 -- the residual-trunk probe, and the only thing left in the plan after P1's readout.
    #
    # P1 measured the geometry loss transferring to the full corpus (Consistency 0.7103 -> 0.8480)
    # and paying for it out of the boundary (1-D_LCFS 0.9603 -> 0.8688) and psi (0.9957 -> 0.9912),
    # which is the wrong direction against a leaderboard whose top five all hold R2_psi >= 0.9988.
    # So R1 turns the geometry term OFF and changes the one thing every charcoal and maroon run has
    # held fixed: the trunk. 149 features -> stem -> 4 residual blocks of 512 -> the same heads,
    # 2.2M parameters against the 2-layer MLP's 80k.
    #
    # The control is M1c, not P1: same corpus, same frozen basis, same 80-shot holdout, geometry off
    # in both, so the only difference is the architecture. GATE: does holdout R2_psi clear 0.998
    # from M1c's 0.99573? A pass says the ceiling was the trunk. A failure says rank 1's edge is
    # input hygiene or ensembling, and 3 h bought that answer before 70 more were spent on it.
    #
    # --early-stop-patience 20 is deliberate: PLATFORM's 0 is what left P1 with best_selection
    # -Infinity and no best-epoch snapshot to fall back on when the machine died.
    "r1": [
        ("R1_residual_geomoff", [*PLATFORM, "--geom-emphasis", "0",
                                 "--arch", "resnet", "--activation", "relu",
                                 "--trunk-width", "512", "--trunk-blocks", "4",
                                 "--trunk-dropout", "0.05",
                                 "--early-stop-patience", "20",
                                 "--epochs", "80", "--log-every", "2"]),
    ],
    # P1 -- bank the platform. Full corpus, geometry at the shipped emphasis, trained to a converged
    # step count. Everything after this is measured against it, so it changes exactly two things
    # against the shipped model: the data (12.7% -> 100%) and the batch (64 -> 256, for throughput).
    # GATE: local S >= 0.970. Below that, the +0.2025 Consistency that emphasis 10 bought at 200k
    # frames does not transfer to the full corpus, and P2 through P5 need re-planning before they run.
    "p1": [
        ("P1_full_e10_conv", [*PLATFORM, "--geom-emphasis", "10", "--geom-warmup", "30",
                              "--epochs", "250", "--log-every", "5"]),
    ],
    "m1": [
        ("M1a_control_200k", [*BASE, "--max-frames", "200000", "--epochs", "420",
                              "--log-every", "60"]),
        ("M1b_mid_600k", [*BASE, "--max-frames", "600000", "--epochs", "140",
                          "--log-every", "20"]),
        # --max-frames 0 = no cap: every frame of all 6,961 training shots.
        ("M1c_full_1541k", [*BASE, "--max-frames", "0", "--epochs", "55",
                            "--log-every", "10"]),
    ],
}

KEEP = ("coefficient targets", "frozen corpus basis", "val split:", "Var(psi)", "epoch ",
        "restored", "S ", "R2_psi", "R2_qb", "1-D_LCFS", "Consistency", "r2_cons_each",
        "Traceback", "Error", "error", "refusing to start", "lambda_lcfs", "L-BFGS",
        "PCA n=", "hidden=", "temporal window", "trunk=", "feature clip",
        # C5's two levers. Without these the chain log cannot say which loss a run used, and
        # "tempered coeff term" is the only line that reports the per-mode weighting it resolved to.
        "tempered coeff term", "selector geometry",
        # C8: without this the chain log cannot say which L_data a run used.
        "edge weight",
        # D2: without these the night log cannot say whether the GRU opened or the freeze held.
        "gate_seq", "freeze-dualctx", "inherited init", "dualseq", "features on host")


def _wait_for_gpu(limit_min: float) -> bool:
    """Block until no other run holds the GPU lock. Liveness is checked through /proc, so a lock
    left behind by a killed run does not stall the queue."""
    from catskills.run_lock import read_locks

    t0 = time.perf_counter()
    announced = False
    while True:
        held = [d for d in read_locks() if d.get("pid") != os.getpid()]
        if not held:
            if announced:
                print(f"### GPU free after {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
            return True
        if (time.perf_counter() - t0) / 60 > limit_min:
            print(f"### still locked after {limit_min:g} min — giving up", flush=True)
            return False
        if not announced:
            o = held[0]
            print(f"### waiting for {o.get('tag')} (pid {o.get('pid')}, "
                  f"{o.get('age_s', 0) / 3600:.1f}h in), limit {limit_min:g} min", flush=True)
            announced = True
        time.sleep(60)


def _publish(tag: str) -> dict | None:
    """Copy the run's score into ``results/maroon/<tag>.json``, where the board reads it.

    The runner writes under ``results/charcoal/`` because it is the charcoal trainer; maroon keeps its
    own directory so the two boards never disagree about which phase a run belongs to.
    """
    src = results_dir() / "charcoal" / f"{tag}.json"
    if not src.is_file():
        src = lp.find_score(tag)
    if not src.is_file():
        print(f"### {tag}: no score json to publish", flush=True)
        return None
    MAROON_DIR.mkdir(parents=True, exist_ok=True)
    dest = MAROON_DIR / f"{tag}.json"
    shutil.copyfile(src, dest)
    try:
        return json.loads(dest.read_text())
    except Exception:
        return None


def _diag(tag: str) -> None:
    """Seen-vs-unseen R2_psi for this arm — the generalisation gap the gate is really about."""
    lab = Path(os.environ.get("CATSKILLS_FUSION_LAB", ""))
    ck = lp.find_checkpoint(tag, art_dir=lab / "artifacts") if lab.name else lp.find_checkpoint(tag)
    if not ck.is_file():
        ck = lp.find_checkpoint(tag)
    if not ck.is_file():
        return
    out = MAROON_DIR / f"psi_diag_{tag}.json"
    print(f"### {tag}: seen/unseen diagnostic", flush=True)
    subprocess.run(
        [sys.executable, "-m", "catskills.scripts.charcoal_psi_diag",
         "--checkpoint", str(ck), "--max-train-shots", "7000", "--eval-shots", "80",
         "--seed", "42", "--seen-shots", "40", "--out", str(out)],
        cwd=ROOT, env={**os.environ, "TQDM_DISABLE": "1", "PYTHONUNBUFFERED": "1"},
        check=False,
    )


def _ensemble(out_tag: str, members: list[str]) -> dict | None:
    """Score the mean of several members as one model, streaming under the chain-log contract.

    Written as a wave entry rather than a hook so the order is visible in WAVES: the ensemble runs
    the moment its members exist, not at the end of the night, and an overnight chain that dies in a
    later arm still leaves this result behind.
    """
    print(f"\n{'#' * 70}\n### {out_tag}\n{'#' * 70}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [sys.executable, "-m", "catskills.scripts.charcoal_ensemble", *members,
         "--out-tag", out_tag, "--max-train-shots", "7000", "--eval-shots", "80", "--seed", "42"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        env={**os.environ, "TQDM_DISABLE": "1", "PYTHONUNBUFFERED": "1"},
    )
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.split("\r")[-1].rstrip()
        if any(k in line for k in KEEP) or "ENSEMBLE" in line or "member" in line:
            print(line, flush=True)
    rc = proc.wait()
    print(f"### {out_tag} exit={rc}  {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    # charcoal_ensemble already writes results/maroon/<out_tag>.json, so there is nothing to publish.
    try:
        return json.loads((MAROON_DIR / f"{out_tag}.json").read_text()) if rc == 0 else None
    except Exception:
        return None


class _Tee:
    """Write everything the chain prints to the board's log as well as to stdout.

    The board reads one file per model (``results/<model>/<model>.log``) and the chain used to have no
    idea that file existed: logging was the launcher's job, so every wave was started with its output
    redirected to /tmp and a ``tail -F`` bridge appending that into the board's log. Every wave since C2
    has needed the bridge rebuilt by hand, and when one was forgotten the board silently showed the
    PREVIOUS wave as the live run -- which is worse than showing nothing, because it looks correct.
    Owning the log here removes the bridge, and with it that whole failure mode.

    Line-buffered and flushed per write: the board polls this file, so a buffered tail would make a
    running phase look stalled.
    """

    def __init__(self, stream, path: Path):
        self._stream = stream
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", buffering=1, encoding="utf-8", errors="replace")

    def write(self, s: str) -> int:
        n = self._stream.write(s)
        try:
            self._fh.write(s)
        except Exception:
            pass  # a full or unmounted lab disk must not kill the chain
        return n

    def flush(self) -> None:
        self._stream.flush()
        try:
            self._fh.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        return False

    def __getattr__(self, name):
        return getattr(self._stream, name)


def chain_log() -> Path:
    """The file the board polls. Derived from MAROON_DIR so purple's override moves it too."""
    return MAROON_DIR / f"{MAROON_DIR.name}.log"


def main() -> int:
    sys.stdout = _Tee(sys.stdout, chain_log())
    sys.stderr = sys.stdout
    names = [a for a in sys.argv[1:] if not a.replace(".", "", 1).isdigit()] or ["m1"]
    unknown = [n for n in names if n not in WAVES]
    if unknown:
        print(f"unknown wave(s) {unknown}; "
              f"usage: maroon_ladder [{'|'.join(WAVES)}]... [wait_minutes]", file=sys.stderr)
        return 1
    nums = [a for a in sys.argv[1:] if a.replace(".", "", 1).isdigit()]
    wait_min = float(nums[0]) if nums else 0.0
    # Several waves chain into one queue so a night's work is one process and one log.
    wave = "+".join(names)
    plan = [entry for n in names for entry in WAVES[n]]

    MAROON_DIR.mkdir(parents=True, exist_ok=True)
    gate = M0_GATE
    if not gate.is_file():
        print("### M0 has not passed — run catskills.scripts.maroon_m0_cache first", flush=True)
        return 3
    if not (json.loads(gate.read_text()).get("gate") or {}).get("passed"):
        print("### M0 gate FAILED — the coefficient cache is not lossless; stopping", flush=True)
        return 3

    summary: list[dict] = []
    for tag, extra in plan:
        if tag.startswith("@ensemble:"):
            out_tag = tag.split(":", 1)[1]
            sc = _ensemble(out_tag, list(extra))
            if sc:
                summary.append({"tag": out_tag, "minutes": 0, **{
                    k: sc.get(k) for k in
                    ("S", "r2_psi", "r2_qb", "one_minus_dlcfs", "consistency", "n_train")
                }})
            continue
        if wait_min > 0 and not _wait_for_gpu(wait_min):
            return 2
        print(f"\n{'#' * 70}\n### {tag}\n{'#' * 70}", flush=True)
        t0 = time.perf_counter()
        proc = subprocess.Popen(
            [sys.executable, "-m", "catskills.run_crc_feq_pinn_charcoal", *extra, "--tag", tag],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            env={**os.environ, "TQDM_DISABLE": "1", "PYTHONUNBUFFERED": "1"},
        )
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.split("\r")[-1].rstrip()
            if any(k in line for k in KEEP):
                print(line, flush=True)
        rc = proc.wait()
        mins = (time.perf_counter() - t0) / 60
        print(f"### {tag} exit={rc}  {mins:.1f} min", flush=True)
        if rc == 0:
            sc = _publish(tag)
            if sc:
                summary.append({"tag": tag, "minutes": round(mins, 1), **{
                    k: sc.get(k) for k in
                    ("S", "r2_psi", "r2_qb", "one_minus_dlcfs", "consistency", "n_train")
                }})
            _diag(tag)
        elif rc == 2:
            print("### lock held by another run — stopping the chain", flush=True)
            return 2

    if summary:
        print(f"\n### {wave.upper()} summary", flush=True)
        print(f"{'tag':22}{'shots':>7}{'S':>10}{'R2_psi':>10}{'Cons':>9}{'min':>7}", flush=True)
        for r in summary:
            print(f"{r['tag']:22}{r.get('n_train') or 0:>7}{r.get('S') or 0:>10.5f}"
                  f"{r.get('r2_psi') or 0:>10.5f}{r.get('consistency') or 0:>9.4f}"
                  f"{r['minutes']:>7.0f}", flush=True)
        (MAROON_DIR / f"{wave}_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    print("\nALLDONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
