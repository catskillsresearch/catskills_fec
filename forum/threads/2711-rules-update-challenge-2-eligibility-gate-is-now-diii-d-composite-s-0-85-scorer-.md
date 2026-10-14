# Rules update — Challenge 2 eligibility gate is now DIII-D composite S ≥ 0.85 (scorer-enforced)

- Thread: https://www.codabench.org/forums/17429/2711/
- Started by: sophelio
- Date created: Aug 18, 2026
- Pinned: yes
- Posts: 1

---

## Post 1 — sophelio

*Aug. 18, 2026, 7:38 p.m.*

A participant raised a fair concern (starter-kit issue #7 on GitHub): the Challenge 2 ratio `G_ratio = S_MAST / S_DIII-D` could be inflated by deliberately weakening a DIII-D entry. The old admissibility gate, `R²ψ > 0.6` on DIII-D, constrained only the flux term — 55% of the composite — so driving `S_DIII-D` down to the eligibility floor bought roughly a 3× ratio boost without breaking any rule.

**Effective today, only entries with a DIII-D composite `S ≥ 0.85` are ranked on Challenge 2.**

What this means in practice:

- **The ratio stays.** Challenge 2 is about zero-shot transfer: do genuinely well on DIII-D, then retain that performance on MAST. Gating the full composite bounds the entire denominator, capping the sandbagging headroom at roughly 1.1× instead of 3×.
- **The scorer now enforces the gate** (v3.2.0): an entry with `S_DIII-D < 0.85` shows `G_ratio = 0` in the "Ch2: G_ratio" column, and its Detailed Results state why.
- **Nothing to do if you are at or above the gate.** For `S_DIII-D ≥ 0.85` the gated and ungated ratios are identical, so existing scores are unchanged and no resubmission is needed.
- **Below-gate entries** keep their currently displayed ratio until their next submission is scored, but Award #2 will be decided under the new rule either way.
- **Challenge 1 is unaffected.**

The new gate is strictly stronger than the old one (`S ≥ 0.85` implies `R²ψ ≥ 0.727` on DIII-D). Full discussion and rationale: https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/7 — thanks to the participant who reported it. The Evaluation and Rules pages here, the starter kit, and the challenge website have all been updated.
