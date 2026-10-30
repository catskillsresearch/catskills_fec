## Phase 3: Reproduce

| Phase | Name | Role |
|---|---|---|
| 1 | Development | Public fold / Codabench Dev |
| 2 | Final | Blind private fold (award scores) |
| 3 | **Reproduce** | Organizer rebuild / internal review (`final.tt`) |

**Deadline:** 31 October 2026 23:59 CDT.  
**Packages required for:** Codabench **1003664** and **EM-catskills-3** (`9db5e666…`).

**Tolerances:** retrain composite $S$ within **0.01** of submitted; shipped weights within **1e-4**.

**Post-hoc package rule:** Phase-3 docs and entrypoints may be finalized after Phase-2 submit; HF prediction blobs are not rewritten — regenerate from Final inputs + weights / retrain.

Standing rebuild path matches **9db5e666** (train-discovered ZGRID freeze equivalent to Dev ±1.2). Submission **1003664** used a mistaken narrow ZGRID (−0.6…+0.8); use `--legacy-zgrid-1003664` only for forensics.
