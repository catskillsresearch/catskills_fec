import time
from concurrent.futures import ProcessPoolExecutor
from src.feq.data import load_shot, shot_paths
from osf.d3d_0_0_13_osf_0_0_12 import cal_at, _solve_one_frame
from osf.osf_0_0_1 import plasma_green, coil_unit_maps, _as1

paths = shot_paths('diii_d_train')
row = load_shot(paths[0])
maps = coil_unit_maps(row)
r, z = row['efit_grid_R'], row['efit_grid_Z']
green = plasma_green(r, z)
t_dst = _as1(row['efit_times'])
n = len(t_dst)

def run_frame(idx):
    cal = cal_at(float(t_dst[idx]))
    return idx, _solve_one_frame(row, idx, cal, maps=maps, green=green)

if __name__ == '__main__':
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=14) as pool:
        results = list(pool.map(run_frame, range(n)))
    t1 = time.time()
    print(f"Parallel solve of {n} frames took {t1 - t0:.2f} s on 14 workers!")
