#!/usr/bin/env python3
"""CRC_FEQ_purple phase chains — one run at a time on one GPU.

    poetry run python -m catskills.scripts.purple_ladder c1 [wait_minutes]
    poetry run python -m catskills.scripts.purple_ladder c2 c3 [wait_minutes]

Purple asks **one** question: were the three temporal-context failures caused by the information or by
the shared stem? See MODEL_DESCRIPTION_CRC_FEQ_purple.md §3. The arms below are single-variable against
maroon's R7 (S 0.97998), which is the same trunk with the context branch absent, and against R10a
(0.97642), which is the same information through the stem purple removes.

The chain mechanics — GPU lock waiting, the ``### <tag>`` log contract the board parses, score
publication, the seen/unseen diagnostic, ensemble entries, the summary table — are maroon's, imported
rather than copied so the two chains cannot drift apart. Only three things are redirected: the results
directory (``results/purple``), the wave table, and the phase configs. The M0 receipt deliberately stays
pointed at maroon's, because purple reuses that *same verified* frozen k=60 basis and re-gating a cache
we already proved lossless at R2_psi 0.99999996 would measure nothing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Claim purple's artifact namespace BEFORE anything reads it: lab_paths resolves MODEL_ID at import,
# and subprocesses inherit this, so every checkpoint and score C2 onward writes is named
# CRC_FEQ_purple_* instead of inheriting the trainer module's charcoal id. Reads still fall back to
# the lineage prefix (lab_paths.find_checkpoint), which is what lets C4 mix maroon's members and what
# keeps C1 — launched before this existed, so written as CRC_FEQ_charcoal_C1_dualctx_probe.pt —
# resolvable by tag. Set rather than overwritten, so an explicit id from the caller still wins.
os.environ.setdefault("CRC_FEQ_MODEL_ID", "CRC_FEQ_purple")

from catskills.lab_paths import results_dir  # noqa: E402
from catskills.scripts import maroon_ladder as ml  # noqa: E402

PURPLE_DIR = results_dir() / "purple"

# R7's platform exactly, which is also what R9 confirmed and what R10a/R10b ran. The warmup and clip
# are not optional: they are the whole difference between R5 diverging at 17M parameters and R7
# converging at the same size.
STABLE_17M = [*ml.PLATFORM, *ml.STABLE, "--trunk-width", "1024", "--trunk-blocks", "8"]

# The context branch. --arch comes AFTER ml.STABLE's "--arch resnet" so argparse takes this one.
# window 5 + smoothdiff is R10a's input exactly, so C1 minus the fusion IS R10a: the comparison
# isolates the stem and nothing else.
DUALCTX = ["--arch", "dualctx", "--ctx-width", "512", "--ctx-blocks", "3",
           "--temporal-window", "5", "--temporal-mode", "smoothdiff"]

# C2's verdict, and what C3/C4 inherit. C2b (window 9) is the best single model of the project at
# S 0.98214; C2a (context encoder 1024x6, 6x the parameters) came in 0.00017 BELOW C1. So the context
# branch wants a longer window, not a bigger encoder -- the same shape of answer R9 gave for the trunk,
# one level down. DUALCTX above is deliberately left at C1's window 5 so the C1/C2 comparison in the
# doc stays reproducible from this file.
DUALCTX_WIN9 = ["--arch", "dualctx", "--ctx-width", "512", "--ctx-blocks", "3",
                "--temporal-window", "9", "--temporal-mode", "smoothdiff"]

# What C4's members are built on. Its own line because C4 is paused pending C5, and "rebuild the pool on
# whichever base wins" then means editing exactly this list rather than eight member configs -- which is
# how C4g/C4h came to disagree with the board's tag list the last time the members were revised by hand.
# Append C5's winning flags here (e.g. the --lambda-coeff-white trio, or --val-geom-frames 20000) if C5
# clears its gate; leave it as C2b's platform if C5 does not.
C4_BASE = [*DUALCTX_WIN9]

# C7's two knobs, kept here rather than inline because they are the only things that change between a
# clamp-12 ladder and a clamp-8 one, and getting them out of step is how an arm ends up measuring two
# variables. C7_LAM_3X is the lambda that reproduces C6's term strength AT THIS CLAMP -- the ladder then
# climbs 3x/9x/27x from it. Both clamps are calibrated to the same term magnitude (lambda =
# 3*0.05*Lw(12)/Lw(clamp) from KOPT_C2b_window9), so switching the pair below switches allocation only:
#
#     clamp 12 -> C7_LAM_3X 0.150   (C6a's arm; ladder 0.45 / 1.35 / 4.05)
#     clamp  8 -> C7_LAM_3X 0.296   (C6b's arm; ladder 0.888 / 2.664 / 7.992)
C7_CLAMP = 12
C7_LAM_3X = 0.150
C7_RUNGS = [round(C7_LAM_3X * m, 3) for m in (3, 9, 27)]


def _c7(lam: float) -> tuple[str, list[str]]:
    """One rung. Everything is C6's arm except lambda, so the wave is a pure dose ladder."""
    return (f"C7_lam{str(lam).replace('.', 'p')}_cl{C7_CLAMP}",
            [*STABLE_17M, *DUALCTX_WIN9, "--epochs", "88", "--val-geom-frames", "20000",
             "--lambda-coeff-white", str(lam), "--coeff-white-power", "1.0",
             "--coeff-white-clamp", str(C7_CLAMP)])

WAVES: dict[str, list[tuple[str, list[str]]]] = {
    # ---------------------------------------------------------------- C1: the decision gate, 2 h
    # One arm. Everything is R7 except that the context block goes to its own encoder and enters the
    # trunk through a scalar gate initialised to ZERO -- so this run STARTS as R7 and the optimizer has
    # to open the gate for context to matter. purple_arch_check.py verifies that claim exactly
    # (gate=0 output is bit-invariant to the context columns).
    #
    # GATE: beat R7's S 0.97998 and cut psi error, not merely trade terms. Beating R10a's 0.97642 is
    # NOT sufficient -- that shows only that fusion beats concatenation, which leaves us where we are.
    # READ: the gate trajectory, logged every epoch as gate=+x.xxxx. A gate decaying to ~0 is the model
    # declining context, and C2 is cancelled rather than run.
    "c1": [
        ("C1_dualctx_probe", [*STABLE_17M, *DUALCTX, "--epochs", "88"]),
    ],
    # ---------------------------------------------------------------- C2: the fusion sweep, 4 h
    # Only if C1 passes. Two single-variable arms off C1.
    # (a) context capacity: does the branch want to be as big as the main trunk?
    # (b) window: a 9-frame window is only affordable once the derivative is a SMOOTHED one, which is
    #     why this was never worth trying before R10a's temporal_mode existed.
    # GATE: beat C1 by more than the +-0.002 single-seed variance measured in maroon §11.8. Smaller is
    # noise and gets rejected, which is the rule R3's mis-specified gate taught us to write down.
    "c2": [
        ("C2a_ctx1024x6", [*STABLE_17M, *DUALCTX, "--epochs", "88",
                           "--ctx-width", "1024", "--ctx-blocks", "6"]),
        ("C2b_window9", [*STABLE_17M, *DUALCTX, "--epochs", "88", "--temporal-window", "9"]),
    ],
    # ---------------------------------------------------------------- C3: the converged base, 6 h
    # C1/C2 run 88 epochs for comparability, but R7 selected epoch 85/100 and R9 epoch 63/64 -- both
    # were still improving when the budget ended, so the winner earns a real schedule. The winner is
    # C2b, so this is DUALCTX_WIN9 and not C1's window 5.
    # GATE: local S >= 0.9860. That is +0.0039 over C2b, which no single arm in this project has ever
    # delivered from schedule alone, so this gate is expected to be MISSED and is written that way on
    # purpose: C3's real job is to hand C4 the best base it can, and the ensemble is what closes the
    # remaining distance. Read it as a measurement of what convergence is worth, not as a pass/fail.
    "c3": [
        ("C3_converged", [*STABLE_17M, *DUALCTX_WIN9, "--epochs", "200",
                          "--early-stop-patience", "25"]),
    ],
    # ---------------------------------------------------------------- C4: the ensemble, 10 h
    # Eight members varying the axes maroon §11.13 established decorrelate, around C3. Two are
    # deliberately CONTEXT-FREE (--arch resnet, window 0): §11.18 measured that the most valuable
    # member is the one whose input representation differs most from the rest of the pool, which is
    # why R10a was worth +0.00035 despite losing 0.0036 on its own. Same data seed throughout, so all
    # members share one holdout and maroon's six shipped checkpoints can join the mean for free.
    # GATE: the 1/N counterfactual at equal count, re-fit on C-series members -- never the previous
    # ensemble, because count alone buys score.
    # PAUSED after C4a, 2026-08-14. The oracle Consistency-vs-k sweep (maroon_oracle_ceiling, k =
    # 10..60) measured that Consistency does NOT saturate at the modes our loss can see: oracle Cons
    # runs 0.9697 at k=20 -> 0.9954 at k=40 -> 0.9972 at k=60, and kappa 0.9448 -> 0.9887. Our best
    # mix sits at Cons 0.947, i.e. between the k=15 and k=20 oracles, so the fit is leaving ~0.027 of
    # Consistency (~0.0055 of S) in modes 21-60 -- 3x what the whole eight-member ensemble was going
    # to buy. Training seven more members on a base that cannot see those modes spends 14 GPU-hours
    # to average a shared blind spot, so the remaining members wait for a base that addresses it.
    #
    # The entries below are the ones still to train. C4a_base is trained and stays out of this list
    # because the chain runs every entry unconditionally -- there is no skip-if-trained -- so leaving
    # it here would silently retrain it on resume. It is still named in the ensemble entry, which
    # resolves members by tag through lab_paths.find_checkpoint. Its exact config, kept so the member
    # stays reproducible now that it is no longer a live entry:
    #
    #   ("C4a_base", [*STABLE_17M, *DUALCTX_WIN9, "--epochs", "88", "--init-seed", "101"])
    "c4": [
        ("C4b_deep10", [*STABLE_17M, *C4_BASE, "--epochs", "88", "--trunk-blocks", "10",
                        "--init-seed", "202"]),
        ("C4c_clip5", [*STABLE_17M, *C4_BASE, "--epochs", "88", "--feature-clip", "5",
                       "--init-seed", "303"]),
        ("C4d_drop15", [*STABLE_17M, *C4_BASE, "--epochs", "88", "--trunk-dropout", "0.15",
                        "--init-seed", "404"]),
        # Batch changes the optimizer's noise rather than the function class, i.e. which minimum the
        # run falls into, and it is free on this card (2.9 of 12 GB in use at 97% GPU). lr is
        # sqrt-scaled with it, or "different noise" would be confounded with "less training".
        ("C4e_batch1024", [*STABLE_17M, *C4_BASE, "--epochs", "88", "--batch-size", "1024",
                           "--lr", "2e-3", "--init-seed", "505"]),
        # C2a's axis. It LOST as a single model (0.00017 under C1) and is nevertheless a member of the
        # best mix measured to date, which is §11.18's lesson arriving a second time.
        ("C4f_ctxwide", [*STABLE_17M, *C4_BASE, "--epochs", "88", "--ctx-width", "1024",
                         "--init-seed", "606"]),
        # These two were "--arch resnet, no context at all", for representation diversity. They are now
        # dualctx instead, but the reason is narrower than it first looked and the first version of this
        # comment overstated it. What is measured (2026-08-14, free combo search, no training):
        #
        #   pure dualctx scales as S(N) = 0.98845 - 0.00661/N, fit on N=1/2/3 at R2 1.00000
        #   trio + R7   (17.0M noctx, S 0.97998)  0.98662   +0.00037 vs the trio's 0.98625
        #   trio + R2b  (2.2M  noctx, S 0.97698)  0.98616   -0.00009
        #   trio + R10a (17.0M shared-stem ctx)   0.98577   -0.00048
        #   a 4th DUALCTX member, from the curve   0.98680   <- the honest counterfactual
        #
        # So a STRONG context-free member does help (R7 is worth +0.00037), just less than another
        # dualctx member would be (-0.00018 against the counterfactual). Weak members are neutral to
        # harmful. The nine-member mix lost to the trio (0.98609 vs 0.98625) because it bought R7's
        # decorrelation and paid for it with five weaker members. That makes the training slots worth
        # spending on dualctx and nothing else -- but R7/R9 already exist, cost zero, and get re-tested
        # by C4's own combo search, which is where the question belongs.
        ("C4g_win13", [*STABLE_17M, *C4_BASE, "--epochs", "88", "--temporal-window", "13",
                       "--init-seed", "707"]),
        ("C4h_win5_ctx256", [*STABLE_17M, *C4_BASE, "--epochs", "88",
                             "--temporal-window", "5", "--ctx-width", "256", "--ctx-blocks", "2",
                             "--init-seed", "808"]),
        # C1/C2a/C2b join for free -- same data seed, same holdout -- so the mean is over TWELVE members
        # without training three of them. The gate is the 1/N counterfactual at equal count.
        ("@ensemble:C4_ensemble", [
            "C3_converged", "C4a_base", "C4b_deep10", "C4c_clip5", "C4d_drop15",
            "C4e_batch1024", "C4f_ctxwide", "C4g_win13", "C4h_win5_ctx256",
            "C1_dualctx_probe", "C2a_ctx1024x6", "C2b_window9",
        ]),
    ],
    # ---------------------------------------------------------------- C5: where the fit is spent, 4 h
    # Two single-variable arms against C2b, the best single model of the project (S 0.98214, Cons
    # 0.92619). Both come from the same measurement, made 2026-08-14 with no training.
    #
    # THE MEASUREMENT. The oracle sweep put ground-truth psi through the first k components of the
    # frozen basis and scored it. Consistency does not saturate where the loss can see:
    #
    #     k      10      15      20      25      30      40      60
    #     Cons   0.7673  0.9390  0.9697  0.9869  0.9912  0.9954  0.9972
    #     kappa  0.4993  0.8593  0.9448  0.9643  0.9679  0.9827  0.9887
    #     R2psi  0.99891 0.99982 0.99996 0.99999 1.00000 1.00000 1.00000
    #
    # Read the last two rows together: psi is essentially exact by k=20 while Consistency still has
    # 0.027 to go. Modes 13-30 alone carry +0.052 of oracle Consistency. So the score's Consistency
    # term lives precisely where psi R2 has stopped being able to see anything -- and our loss is
    # psi-shaped in both of its flux terms.
    #
    # Then maroon_kopt on C2b measured where the gradient actually goes, and where the model actually
    # is, per mode (18,573 holdout frames):
    #
    #     band            1-12    13-30   31-60
    #     psi error       93.68%   6.18%   0.14%   <- and therefore the gradient share, since the
    #     model R2_j      0.978    0.764   0.429      basis is orthonormal and L_coeff is unwhitened
    #
    # Half the basis receives 0.14% of the gradient. Note what this does NOT say: zero of 60
    # components is predicted worse than the basis mean, so the high modes are not unlearned, they are
    # HALF learned (R2 0.43) on almost no signal. That also corrects maroon 11.17, which measured 14
    # hurting components on R7 -- the dualctx models fixed that, so the argument against whitening
    # ("it pours capacity into noise modes") no longer has evidence behind it at this base.
    #
    # THE CORRECTION, made 2026-08-14 after 12 minutes of the first C5a. The oracle curve above says
    # truncating at k=20 costs 0.027 of Consistency, and the first version of this arm read that as
    # "the deficit is in modes 21-60" and clamped at mode 25. That inference is wrong, and purple_band_probe
    # measured why: truncation sets a mode to the corpus MEAN, whereas our model gets it partly right
    # (R2 0.43 averaged over modes 31-60), so the oracle overstates what those modes cost US. Handing
    # C2b the TRUE coefficients one band at a time, on the same 80 shots, prices each band directly:
    #
    #     band     Consistency recovered      share of the 0.07101 recoverable
    #     1-12     +0.04100                   59.4%
    #     13-20    +0.01653                   24.0%
    #     21-30    +0.00963                   14.0%
    #     31-60    +0.00180                    2.6%
    #
    # So 83% of the recoverable Consistency is in modes 1-20 -- the modes that already receive 98.7% of
    # the gradient -- and the tail the first arm aimed at is worth 2.6%. Clamp 25 would have put 61% of
    # the new term on bands worth 17% while cutting the 59%-band's share to 10%. Its live trace agreed:
    # val_Cons fell 0.7012 -> 0.6170 -> 0.5296 over epochs 1-8 while val_psi improved.
    #
    # (a) C5a: reallocate the coefficient gradient WITHIN the modes that matter. Divide each mode's
    #     residual by that mode's corpus std, floored at mode 12's. That gives 60.4% / 31.4% / 7.4% /
    #     0.9% across the four bands, against a measured value distribution of 59.4 / 24.0 / 14.0 /
    #     2.6 -- the closest match available from this one-parameter family. What it actually fixes is
    #     WITHIN the low band: mode 1 alone is 57% of coefficient error at R2 0.9993, so it monopolises
    #     the gradient while modes 6-12 sit at R2 0.93-0.97 and get almost none. L_data and L_coeff keep
    #     full weight, so the low modes keep their absolute gradient and only stop being dominated by
    #     mode 1. lambda 0.05 sizes the new term at 0.25*L_coeff at C2b's converged error.
    # (b) C5b: fix the selector. Every run since R1 selected weights on -val_mse, because val_cons is
    #     only finite when the geometry loss is on and these runs all have it off; their logs print
    #     "best selection=-0.0000". --val-geom-frames derives geometry targets on 20k held-out frames
    #     for the SELECTOR only, with no change to the loss, so selection sees 0.55*R2psi + 0.20*Cons.
    #     C3 is why this is not cosmetic: it selected an epoch whose psi MSE was the project's best
    #     and whose Consistency was 0.00254 worse.
    #
    # THE DESIGN. Both arms carry the selector fix, and the two subtractions are what isolate each
    # change: C5b - C2b is the selector, C5a - C5b is the loss. Running the loss arm with C2b's
    # psi-MSE selector instead would have confounded it in the WORSE direction -- if whitening trades
    # a little psi for Consistency, a psi-MSE selector picks the least-Consistent epoch of the run and
    # the loss change gets blamed for the selector's choice. Two runs, both questions answered.
    #
    # GATE (both): Consistency above C2b's 0.92619 AND official R2psi at or above C2b's 0.99926.
    # Buying Consistency by giving up psi is the P1 trade that already failed and is a fail here.
    # MARGIN: C4a is C2b's config at init-seed 101 and scored Cons 0.92471 against C2b's 0.92619, so
    # the same-config seed spread on THIS term is 0.0015 -- 3x the 0.0005 the C2 note assumed from
    # R7/R9. A pass needs Cons >= ~0.9285, not merely above 0.92619.
    "c5": [
        ("C5a_white12", [*STABLE_17M, *DUALCTX_WIN9, "--epochs", "88",
                         "--val-geom-frames", "20000",
                         "--lambda-coeff-white", "0.05", "--coeff-white-power", "1.0",
                         "--coeff-white-clamp", "12"]),
        ("C5b_selcons", [*STABLE_17M, *DUALCTX_WIN9, "--epochs", "88",
                         "--val-geom-frames", "20000"]),
    ],
    # ---------------------------------------------------------------- C6: how much of the lever, 6 h
    # C5's result, 2026-08-14, both arms against C2b (S 0.98214, Cons 0.92619, R2psi 0.99926):
    #
    #     C5a_white12  S 0.98407  Cons 0.93519  R2psi 0.999282   +0.00193 S, +0.0090 Cons
    #     C5b_selcons  S 0.98206  Cons 0.92584  R2psi 0.999260   -0.00008 S, -0.0004 Cons
    #
    # Read the two subtractions the design was built to give. C5b - C2b is the selector alone: it moves
    # nothing, at 1/4 of the same-config seed spread, so the Consistency-aware selector is measured dead
    # and is kept only because it costs nothing and makes val_cons readable. C5a - C5b is therefore the
    # whole +0.0090, and it belongs to the loss. It is also the first change in this project to buy
    # Consistency without paying psi for it: R2psi went UP by 0.00002.
    #
    # So the lever works and C6 asks the only two questions left about it -- how hard to pull it, and
    # whether the band shares are what is doing the work.
    #
    # THE CALIBRATION. Every arm below carries the SAME term magnitude, 3x C5a's. The clamp changes the
    # divisor floor and therefore changes L_coeff_w by a factor of 8 across these three settings, so
    # holding lambda fixed while moving the clamp would confound "where the gradient goes" with "how much
    # of it there is" -- exactly the confound C5's two-arm design was built to avoid one level up. lambda
    # is re-derived per clamp from C2b's measured per-mode error and target variance (KOPT_C2b_window9:
    # lambda = 3 * 0.05 * Lw(12) / Lw(clamp)), which equalises the term at 0.00132 for all three:
    #
    #     clamp   sd floor   lambda    gradient share 1-12 / 13-20 / 21-30 / 31-60
    #        8      0.3348    0.296     76.6   18.6    4.4    0.5
    #       12      0.1835    0.150     60.4   31.4    7.4    0.9    <- C5a's shares, tripled
    #       20      0.0383    0.036     14.5   40.1   40.6    4.7
    #     (measured VALUE of each band, from purple_band_probe:  59.4 / 24.0 / 14.0 / 2.6)
    #
    # (a) C6a: C5a's allocation at 3x strength. Answers "was 0.05 the right size?" and nothing else.
    # (b) C6b: clamp 8 pushes the shares back TOWARD unwhitened L_coeff (93.7 / 6.2 / 0.1). If this ties
    #     C6a then the mechanism is not the reallocation and the term is acting as plain extra coefficient
    #     weight, which would mean --lambda-coeff-white is the wrong knob and 0.25*L_coeff should just go up.
    # (c) C6c: clamp 20 overshoots deliberately, putting 45% on bands worth 17%. This is the arm the
    #     band probe predicts should LOSE, and it is here because the first version of C5a was clamped at
    #     25 on exactly that mis-inference and was killed at 12 minutes on its live trace rather than
    #     measured. Cheap to settle it properly: a theory that only ever predicts wins is not being tested.
    #
    # GATE: Cons >= 0.9367 (C5a's 0.93519 plus the 0.0015 same-config seed spread from C4a/C2b) AND
    # official R2psi >= 0.99926. An arm that ties C5a inside 0.0015 is a null, not a win, and the honest
    # reading is then that the lever saturates at lambda 0.05 -- which is still a result, since it makes
    # C5a the base for C4's rebuild and stops us spending GPU-hours climbing a flat curve.
    # RISK: C5a spiked its loss at epoch 5 and recovered; 3x the term is 2.8x the 0.25*L_coeff term, so a
    # divergence here is a real possibility. grad_clip and lr_warmup from ml.STABLE are what make that
    # recoverable, and a diverged arm is itself the answer to "how hard can this be pulled".
    # Whatever wins, all three are also ensemble members on an axis -- loss shape -- that nothing else in
    # the pool varies, so C4's combo search gets three genuinely new candidates out of this either way.
    "c6": [
        ("C6a_white12_lam15", [*STABLE_17M, *DUALCTX_WIN9, "--epochs", "88",
                               "--val-geom-frames", "20000",
                               "--lambda-coeff-white", "0.150", "--coeff-white-power", "1.0",
                               "--coeff-white-clamp", "12"]),
        ("C6b_white8_lam296", [*STABLE_17M, *DUALCTX_WIN9, "--epochs", "88",
                               "--val-geom-frames", "20000",
                               "--lambda-coeff-white", "0.296", "--coeff-white-power", "1.0",
                               "--coeff-white-clamp", "8"]),
        ("C6c_white20_lam36", [*STABLE_17M, *DUALCTX_WIN9, "--epochs", "88",
                               "--val-geom-frames", "20000",
                               "--lambda-coeff-white", "0.036", "--coeff-white-power", "1.0",
                               "--coeff-white-clamp", "20"]),
    ],
    # ---------------------------------------------------------------- C7: the dose ladder, 5.5 h
    # WHY THIS AND NOT THE C4 REBUILD. The 2026-08-14 board pushed us from 6th to 8th, and the term-wise
    # decomposition says the same thing against every rank above us: we are TIED with rank 1 on R2psi
    # (0.9994) and hold the BEST q95/betaN on the board (+0.0006 in our favour), and Consistency is
    # 91-95% of the entire shortfall. So the only lever worth GPU time is the one that moves Consistency,
    # and C5/C6 turned that into a measured dose-response curve on the official 80-shot holdout:
    #
    #     lambda   0     0.05    0.15
    #     Cons     0.92619  0.93519  0.94190      +0.0061 of Consistency per 3x lambda
    #     R2psi    0.999263 0.999282 0.999209     -0.00007 of psi per 3x lambda
    #
    # That is ~17 units of Consistency per unit of psi, against score weights of 0.20 and 0.55 -- the
    # trade is favourable by 6x and has not bent yet. Extrapolated with the measured +0.0141 ensemble
    # lift, lambda 1.35 projects to public 0.990 (5th) and 4.05 to 0.991 (2nd). It is a two-point fit
    # extrapolated fourfold and it WILL break somewhere; the point of this wave is to find where, for
    # 2 h per point. Against that, the eight C4 members are 16 GPU-hours to buy the +0.0141 lift that on
    # its own lands at 7th, and base quality multiplies through an ensemble while an ensemble does not
    # multiply through itself. C4 therefore waits for the base this wave names.
    #
    # CLAMP. Set from C6b: it and C6a carry identical term strength and differ only in allocation, so
    # whichever wins is the allocation this ladder should climb. C7_CLAMP/C7_LAM_3X are the two numbers
    # to edit, and the arms below derive their lambdas from them, so the wave stays a pure dose ladder.
    # Ordered cheapest-risk first -- the top of this ladder is where divergence is expected, and a crash
    # at 81x must still leave 9x and 27x trained.
    #
    # GATE: Consistency above the arm below it by more than the 0.0015 same-config seed spread, AND
    # R2psi >= 0.9985. The second number is not C6's 0.99926: psi is EXPECTED to be spent here and the
    # question is the exchange rate, so the bar is where spending it stops being worth it. At weights
    # 0.55/0.20, giving up 0.001 of psi needs +0.00275 of Consistency to break even, which is well
    # inside what one rung buys. Below 0.9985 the trade has gone bad and the ladder stops.
    # READ: the exchange rate per rung, not the absolute score. Two rungs that both pass but at a
    # collapsing rate mean the curve has bent and the remaining budget belongs to C4 after all.
    "c7": [_c7(lam) for lam in C7_RUNGS],
    # ---------------------------------------------------------------- C8: the edge-weighted L_data strike, 2 h
    # Architecture is LOCKED. Rank 1's method string is the family we are already in (PCA + dual-context
    # ResNet flux ensemble). We hold the best public R2psi (0.9996) and R2qb (0.9949) on the board; the
    # remaining gap is Consistency (+0.0155 to rank 1) and D_LCFS. C7 proved the PCA-space lever
    # saturates at lambda 1.35. C8 changes WHERE on the 65x65 grid L_data is spent, not the trunk.
    #
    # C7b is the core-dominant member (uniform MSE + whitened coeff at the last dose that still moved
    # Consistency). C8a is the edge-dominant twin: same flags, L_data reweighted by |∇ψ_gt|^1 with
    # mix 0.6. Weights are mean-normalized so lambda 1.35 stays calibrated. C8b (donut on psi_n) is
    # coded and queued; it does not run unless C8a misses and the band probe still says the deficit
    # is spatial rather than modal.
    #
    # After C8a scores, the free holdout sweep is C7b+C8a at 1:1 and 2:1 (duplicate C7b in the mean)
    # plus the existing 8-mix with C8a added. Package only if projected public >= 0.9907 (#4 bar).
    # ARM GATE vs C7b (S 0.98884 / R2psi 0.999360 / Cons 0.95773 / 1-D 0.98577):
    #   Cons +> 0.0015, R2psi >= 0.9993, 1-D_LCFS not down more than 0.002.
    # A psi-only gain with Cons flat is a miss.
    "c8": [
        ("C8a_edgegrad",
         [*STABLE_17M, *DUALCTX_WIN9, "--epochs", "88", "--val-geom-frames", "20000",
          "--lambda-coeff-white", "1.35", "--coeff-white-power", "1.0",
          "--coeff-white-clamp", "12",
          "--edge-weight", "grad", "--edge-mix", "0.6", "--edge-power", "1.0"]),
    ],
    "c8b": [
        ("C8b_edgeband",
         [*STABLE_17M, *DUALCTX_WIN9, "--epochs", "88", "--val-geom-frames", "20000",
          "--lambda-coeff-white", "1.35", "--coeff-white-power", "1.0",
          "--coeff-white-clamp", "12",
          "--edge-weight", "band", "--edge-mix", "0.6",
          "--edge-band-lo", "0.70", "--edge-band-hi", "1.05"]),
    ],
    # ---------------------------------------------------------------- C9: small geometry on C7b, 2 h
    # C8a missed: |∇ψ|-weighted L_data bought Cons +0.00035. The missing Consistency is not underfit
    # edge pixels. P1 closed geometry at emphasis 10 on a weak map (paid 1-D -0.092). This is a fifth
    # of that dose on the 0.99936 map, deficit-weighted so tri_top does not eat the gradient, annealed
    # to 0.5 over the last quarter so flux can finish. LCFS stays off. C8b is not this wave.
    #
    # ARM GATE vs C7b: Cons +> 0.0015, R2psi >= 0.9993, 1-D not down more than 0.002.
    # A Cons gain paid in D_LCFS is a miss. If Cons +0.0005..0.0015 and boundary holds, climb to
    # emphasis 5. If Cons +< 0.0005, null -> C10 (lambda-h1). If 1-D drops > 0.002, stop.
    "c9": [
        ("C9a_geom2",
         [*STABLE_17M, *DUALCTX_WIN9, "--epochs", "88", "--val-geom-frames", "20000",
          "--lambda-coeff-white", "1.35", "--coeff-white-power", "1.0",
          "--coeff-white-clamp", "12",
          "--geom-emphasis", "2", "--geom-weights", "deficit",
          "--geom-emphasis-final", "0.5", "--geom-anneal-tail", "0.25"]),
    ],
    # ---------------------------------------------------------------- D1: the trunk reopens, 2 h
    # The architecture lock rested on rank 1's method string being our own family. The 2026-08-15
    # 19:58 board falsified it: the new #1 (pshishkin, 0.9932) declares 'seq + mlp' WITH the
    # compute-light badge, and beats us on all four terms (Cons is 80% of the weighted gap).
    # Context has now been delivered three ways: flat concat lost three times (R2a/R4/R10a), a gated
    # second encoder won (C1) and saturated (C7/C8/C9), and a sequence encoder is the one delivery
    # never tried. D1a is a small GRU (2x256, ~0.7M params) over the last 9 RAW frames — temporal
    # mode 'stack' lays out [x_{t-4}..x_{t+4}] per frame, shot-clamped, NOT [raw|avg|diff] — with
    # the usual heads on the final hidden state. Same frozen k=60 basis, same packager contract via
    # the checkpoint's arch key. NO lambda-coeff-white, NO geom, NO edge-weight: the arm isolates
    # the trunk. --val-geom-frames stays on (selector only, measured dead as a lever, free, and it
    # makes val_Cons readable). STABLE_17M's trunk-width/blocks are inert on this arch.
    #
    # ARM GATE vs C2b (dualctx window 9, no whitening: S 0.98214 / Cons 0.92619):
    #   MISS (stop the family): Cons within 0.0015 of 0.92619 AND S below 0.98214 — concatenated
    #     context failing in a new costume.
    #   MEMBER: Cons >= 0.935 (C5a territory) -> free mix C7b+C8a+C9a+D1a; package only if
    #     projected public >= 0.9918 under the C9mix CALIB (public ≈ local − 0.0006).
    #   PASS (climb): S >= 0.986 or Cons >= 0.950 -> D1b is this trunk plus C7b's lambda 1.35,
    #     NOT a bigger GRU. A full-shot recurrence is the ONE allowed engineering variant if this
    #     misses on mechanics rather than physics; nothing else this weekend.
    "d1": [
        ("D1a_seq",
         [*STABLE_17M, "--arch", "seq", "--temporal-window", "9", "--temporal-mode", "stack",
          "--seq-hidden", "256", "--seq-layers", "2",
          "--epochs", "88", "--val-geom-frames", "20000"]),
    ],
}


def main() -> int:
    # Redirect the shared chain at purple's results directory. _publish, _ensemble, _diag and the
    # summary all read this one module global, so this is the whole redirection.
    ml.MAROON_DIR = PURPLE_DIR
    ml.WAVES = WAVES
    PURPLE_DIR.mkdir(parents=True, exist_ok=True)
    if not sys.argv[1:] or all(a.replace(".", "", 1).isdigit() for a in sys.argv[1:]):
        sys.argv.insert(1, "c1")
    return ml.main()


if __name__ == "__main__":
    raise SystemExit(main())
