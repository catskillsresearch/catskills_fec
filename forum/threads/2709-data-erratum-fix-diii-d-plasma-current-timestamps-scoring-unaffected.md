# Data erratum + fix: DIII-D plasma-current timestamps (scoring unaffected)

- Thread: https://www.codabench.org/forums/17429/2709/
- Started by: sophelio
- Date created: Aug 18, 2026
- Pinned: yes
- Posts: 1

---

## Post 1 — sophelio

*Aug. 18, 2026, 6:59 p.m.*

A participant found that on about 69% of DIII-D shots the *input* column `magnetics_plasma_current_times` carries the wrong time axis, so interpolating Ip onto `efit_times` there reads pre-plasma noise (about 2 kA) instead of the real current (about 1 MA). **The Ip values themselves, all targets, `efit_times`, everything on MAST, and all scoring are unaffected** — and everyone has the same data, so no one has been advantaged or disadvantaged.

A verified drop-in fix now ships in the starter kit and works on the train and test files exactly as they are — no re-download needed:

```python
from data_fixes import fix_d3d_ip_times

ip_times = fix_d3d_ip_times(row)   # per-shot corrected axis
```

The helper decides **per shot** (about 31% of shots ship with the correct axis, so please don't blanket-shift everything), and the baseline in `experiments.py` already applies it.

- Fix + full details: https://github.com/Sophelio/fusion-equilibrium-challenge-starter#data-errata-v110
- Announcement: https://fusion-equilibrium-challenge.sophelio.io/announcements/
- Original report (thanks to pshishkin!): https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/6

A corrected v1.1.1 dataset release will follow and will match the fix's output exactly, so nothing you build now needs rework. A small MAST Thomson channel-alignment fix (`align_mast_thomson_core`, issue #5) ships alongside it.
