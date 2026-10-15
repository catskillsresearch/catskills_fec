#!/usr/bin/env python3
"""Inject DedicatedScalarNet (or blended) q95 & betaN into a DIII-D public test npz.
"""
import argparse
import time
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Replace q95 and betaN in DIII-D npz.")
    parser.add_argument("--base-npz", type=str, required=True, help="Base diii_d_public_test.npz")
    parser.add_argument("--scalars-npz", type=str, required=True, help="Scalars public_test_dedicated_scalars.npz")
    parser.add_argument("--output-npz", type=str, required=True, help="Output diii_d_public_test.npz")
    parser.add_argument("--blend-weight", type=float, default=1.0, help="Weight on new scalars (1.0 = pure new, 0.5 = 50/50 blend)")
    args = parser.parse_args()

    base_path = Path(args.base_npz)
    sc_path = Path(args.scalars_npz)
    out_path = Path(args.output_npz)

    print(f"Loading base: {base_path}...")
    base = np.load(base_path)
    print(f"Loading scalars: {sc_path}...")
    scalars = np.load(sc_path)

    w = args.blend_weight
    new_dict = {}
    q_count = 0
    b_count = 0

    t0 = time.time()
    for k in base.keys():
        if k.endswith("_q95"):
            orig_q = base[k]
            if k in scalars:
                new_q = scalars[k].reshape(orig_q.shape).astype(orig_q.dtype)
                new_dict[k] = (1.0 - w) * orig_q + w * new_q
                q_count += 1
            else:
                new_dict[k] = orig_q
        elif k.endswith("_betaN"):
            orig_b = base[k]
            if k in scalars:
                new_b = scalars[k].reshape(orig_b.shape).astype(orig_b.dtype)
                new_dict[k] = (1.0 - w) * orig_b + w * new_b
                b_count += 1
            else:
                new_dict[k] = orig_b
        else:
            new_dict[k] = base[k]

    print(f"Replaced {q_count} q95 and {b_count} betaN vectors (blend weight: {w:.2f})")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {out_path}...")
    np.savez_compressed(out_path, **new_dict)
    print(f"Done in {time.time() - t0:.2f} s. Size: {out_path.stat().st_size / 1e6:.1f} MB")

if __name__ == "__main__":
    main()
