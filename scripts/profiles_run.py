"""Run all 4,121 calibrated profiles (parallel) and the single-core timing benchmark.

Usage: python scripts/profiles_run.py {all|bench_serial|bench_parallel|tracemalloc}
  all           all policies and rule fits on every profile -> outputs/profiles/profile_results.csv
  bench_serial  deployment pipeline timing on one core      -> outputs/profiles/timing_bench_serial.csv
BLAS threads are pinned to 1 so each process uses one core.
Requires outputs/profiles/series_profiles.csv from scripts/profiles_calibrate.py.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import tracemalloc  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.profiles_core import core, full  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "profiles"
N_WORKERS = os.cpu_count() or 1


def rows():
    prof = pd.read_csv(OUT / "series_profiles.csv")
    return prof.sort_values("q0", ascending=False).to_dict("records")   # big jobs first


def machine():
    def sh(cmd):
        try:
            return subprocess.check_output(cmd, text=True).strip()
        except (OSError, subprocess.CalledProcessError) as e:
            return f"unavailable ({e})"
    return dict(cpu=sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
                logical_cores=os.cpu_count(),
                perf_cores=sh(["sysctl", "-n", "hw.perflevel0.physicalcpu"]),
                eff_cores=sh(["sysctl", "-n", "hw.perflevel1.physicalcpu"]),
                mem_bytes=sh(["sysctl", "-n", "hw.memsize"]),
                platform=platform.platform(), python=platform.python_version(),
                numpy=np.__version__, blas_threads=1)


def run_batch(fn, rs, workers):
    t0 = time.perf_counter()
    if workers == 1:
        res = [fn(r) for r in rs]
    else:
        with Pool(workers) as pool:
            res = list(pool.imap_unordered(fn, rs, chunksize=2))
    wall = time.perf_counter() - t0
    child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return pd.DataFrame(res), dict(wall_s=wall, workers=workers, n=len(rs),
                                   peak_rss_children_bytes=child, peak_rss_parent_bytes=own)


def main(mode):
    OUT.mkdir(parents=True, exist_ok=True)
    rs = rows()
    meta = machine()
    if mode == "all":
        df, info = run_batch(full, rs, N_WORKERS)
        df.to_csv(OUT / "profile_results.csv", index=False)
    elif mode in ("bench_serial", "bench_parallel"):
        workers = 1 if mode == "bench_serial" else N_WORKERS
        df, info = run_batch(core, rs, workers)
        df.to_csv(OUT / f"timing_{mode}.csv", index=False)
    elif mode == "tracemalloc":
        big = rs[0]
        tracemalloc.start()
        core(big)
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        info = dict(store_id=big["store_id"], product_id=big["product_id"], q0=big["q0"],
                    archetype=big["archetype"], tracemalloc_peak_bytes=peak)
        df = None
    else:
        raise SystemExit(f"unknown mode {mode}")
    info = {**info, "machine": meta, "mode": mode}
    (OUT / f"run_{mode}.json").write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps(info, indent=1, default=str))
    if df is not None:
        print(df.filter(like="t_").describe(percentiles=[.5, .95]).round(4).T.to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
