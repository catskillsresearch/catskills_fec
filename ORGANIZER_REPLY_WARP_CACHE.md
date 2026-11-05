# Organizer reply — warp_basis / LS_O1_W1

**To:** FusionEC-26 organizers  
**Re:** `lambda-w > 0 but warp_basis cache missing` at commit `907d045`

---

Hi Ganatma,

You are right: at `907d045`, `psi_tailfit.py` names `warp_coeff_cache` and reads `results/brown/warp_basis/<key>.npz`, but the builder was not in that tree. Sorry — that was an omission in the rebuild packaging.

**Pinned on `main`:** `8c63e5113c20f333e1743c11ecd75c82b18f622f`  
https://github.com/catskillsresearch/catskills_fec/commit/8c63e5113c20f333e1743c11ecd75c82b18f622f  

That tip includes the warp_coeff port (`f384fed`: `catskills/warp_coeff.py`, `catskills/scripts/warp_coeff_cache.py`, and `scripts/phase3_rebuild_all_d3d.sh` updates for `mkdir results/brown`, `maroon_m0_cache`, and `warp_coeff_cache` before `LS_O1_W1`) plus this reply and the longer rebuild guide.

**Prebuilt warp cache (same release as the Olive O1 parent):**

- https://github.com/catskillsresearch/catskills_fec/releases/download/phase3-rebuild-6dc3637/warp_basis_train_seed42.tar.gz (~9 MB; 6,961 train shots)  
- https://github.com/catskillsresearch/catskills_fec/releases/download/phase3-rebuild-6dc3637/README_warp_basis.txt  

```bash
tar -xzf warp_basis_train_seed42.tar.gz -C "$CATSKILLS_FUSION_LAB"
# → results/brown/warp_basis/<key>.npz
```

Or rebuild: `.venv/bin/python -m catskills.scripts.warp_coeff_cache --keys train --workers 12`  
(after `maroon_m0_cache` / `coeffs_v1`; settings: `sample_cache_keys(7080, seed=42)[80:]`, PCA \(k=60\), Thomson on).

Then W1 is unchanged:

```bash
.venv/bin/python -m catskills.scripts.psi_tailfit --parent O1 --power 0.0 --seed 42 \
  --lambda-w 1 --tag LS_O1_W1 --epochs 220 --patience 30
```

`LS_O1_101` does not need the warp cache. Agreed on your other notes: `maroon_m0_cache` (as in `REPRODUCE.md`) and creating `results/brown` — both are now in the rebuild script; `psi_tailfit` also creates `results/brown` before writing.

**Longer rebuild guide (standing reference on `main`):**  
[`ORGANIZER_REPLY_REBUILD.md`](https://github.com/catskillsresearch/catskills_fec/blob/main/ORGANIZER_REPLY_REBUILD.md)  
— full DIII-D member recipes, ensemble/dZ policy, lab commits, MAST data, and the release asset list (Olive `.pt` / report PDF / this warp tarball). Use with code at `8c63e51` or later.

Thanks,  
Lars
