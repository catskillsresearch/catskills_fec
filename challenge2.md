# Catskills Research Company — Challenge 2 (Zero-shot DIII-D → MAST)

**Team:** Catskills Research Company  
**Goal:** Maximize $G_{\mathrm{ratio}}=S_{\mathrm{MAST}}/S_{\mathrm{DIII\text{-}D}}$ with $R^2_\psi>0.6$ on DIII-D — win Award #2  
**Status:** **Deferred** while Challenge 1 is the active focus  
**Companion:** [`challenge1.md`](challenge1.md)

---

## 1. Objective

Train only on DIII-D labels. Apply zero-shot to MAST (`mast_public_test` inputs; **no** `mast_train`). One Codabench submission enters both challenges; Challenge 2 **requires** both machine `.npz` files (DIII-D-only → $G_{\mathrm{ratio}}=0$).

---

## 2. Why Challenge 1 samples are not Challenge 2 solutions

Packaged `ridge_pca` / `mlp_pca` / `pinn_gs` emit MAST files for **format** only. Features still use the **DIII-D 21-coil layout**, zero-filling missing MAST channels. Starter guide reports naive coil transfer ~SSIM 0.83 → ~0.10 on MAST. We have **no** official $G_{\mathrm{ratio}}$ smoke table yet (MAST GT withheld; only public-test inputs + 3 demo parquets).

---

## 3. Planned differentiators (when we resume)

| Lever | Intent |
|-------|--------|
| Synthetic diagnostics | $I_p$, Thomson $p_e\sim n_e T_e$, coil-energy / asymmetry — machine-agnostic |
| Coil geometry rasters | `coil_R/Z` + currents as spatial channels, not named F-coils |
| Training-time GS / PINN | Equilibrium residual so $\hat\psi$ is physics, not DIII-D texture |
| Normalized flux shape | Survive amplitude / sign differences (scorer handles sign only) |
| Neural operators | Longer-horizon FNO / DeepONet on physical $(R,Z)$ |

---

## 4. Relation to Challenge 1

- **Docs / R&D focus:** separable — this file vs `challenge1.md`.  
- **Submission plumbing:** shared Route B zip; Ch2 needs a real MAST head, not zeros-in-coil-slots.  
- **Gate:** Ch2 entries need $R^2_\psi>0.6$ on DIII-D — a strong Challenge 1 model is a prerequisite, not a distraction.

Until Codabench is live and Challenge 1 baselines are scaled, do not burn cycle on MAST transfer experiments beyond feature sketches in `catskills/features.py`.
