CRC_FEQ warp_basis cache (S34.12 Arm A) for psi_tailfit --lambda-w > 0 (LS_O1_W1).

Build settings (matches catskills.scripts.warp_coeff_cache defaults):
  keys: sample_cache_keys(7080, seed=42, include_thomson=True)[80:]  # 6961 train shots
  PCA k=60, include_thomson=True
  workers: 12 (historical)

Each file: results/brown/warp_basis/<cache_key>.npz
  B: float32 (60, 5) orthonormal warp directions
  names: iso, stretch_z, shift_r, shift_z, gain

Extract from $CATSKILLS_FUSION_LAB (lab root):
  tar -xaf warp_basis_train_seed42.tar.gz -C "$CATSKILLS_FUSION_LAB"

Or rebuild:
  .venv/bin/python -m catskills.scripts.warp_coeff_cache --keys train --workers 12

Source commit (builder on catskills_fec): f384fed3a82e2fdf5858b7573dcd1f5521876784
