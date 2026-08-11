"""Parallel ensemble driver: run every realization's .inp, extract flood
metrics, cache to parquet/npz.

Must set KMP_DUPLICATE_LIB_OK before `openswmm.engine` is imported anywhere
in the process -- the engine links Homebrew's libomp while conda-forge's
numpy/scipy stack loads its own copy, and macOS aborts on the duplicate
runtime otherwise. This is the documented, standard workaround for that
exact conda + Homebrew-built-extension conflict, not a numerical hazard.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# 4 workers x 2 OpenMP threads = 8, matching an 8-thread machine; avoids
# oversubscription when the ensemble runs at N_WORKERS parallelism.
os.environ.setdefault("OMP_NUM_THREADS", "2")

import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def run_one(inp_path: Path, realization_id: str) -> dict:
    """Run one realization to completion; return a flat metrics dict and
    write per-node / per-triangle arrays to `<run_dir>/arrays.npz`."""
    from openswmm.engine import Solver, HAS_2D, Statistics, MassBalance

    t0 = time.time()
    with Solver(str(inp_path)) as s:
        if not (HAS_2D and s.surface2d.is_active):
            raise RuntimeError(f"{realization_id}: 2D module not active")
        surf = s.surface2d
        for _ in s.steps():
            pass
        wall_s = time.time() - t0

        mb1d = MassBalance(s)
        stats = Statistics(s)
        mb2d = surf.get_mass_balance()

        node_vol_flooded = np.asarray(stats.node_vol_flooded)
        node_max_depth = np.asarray(stats.node_max_depth)
        node_time_flooded = np.asarray(stats.node_time_flooded)
        node_max_overflow = np.asarray(stats.node_max_overflow)
        node_names = [s.nodes.get_id(i) for i in range(len(node_vol_flooded))]

        tri_max_depth = np.asarray(surf.get_stat_max_depths())
        tri_max_vel = np.asarray(surf.get_stat_max_velocities())
        tri_area = np.array([surf.get_triangle_area(i) for i in range(surf.n_triangles)])

        wet_thresh = config.FLOOD_DEPTH_THRESHOLD_M
        wet = tri_max_depth > wet_thresh
        flood_volume_2d = float(np.sum(tri_max_depth[wet] * tri_area[wet]))

        # instantaneous total_volume time series isn't retained after the
        # run ends; total_volume here is 2D storage at the FINAL step, so
        # "peak flood volume" is taken from the flood *envelope* (per-cell
        # max depth x area), which is well-defined post-run and is what the
        # box/scatter plots in figures.py use throughout.
        metrics = {
            "realization_id": realization_id,
            "wall_s": wall_s,
            "runoff_continuity_error_pct": mb1d.runoff_continuity_error,
            "routing_continuity_error_pct": mb1d.routing_continuity_error,
            "continuity_error_2d": mb2d["continuity_error"],
            "rainfall_in_2d_m3": mb2d["rainfall_in"],
            "coupling_1d_to_2d_in_m3": mb2d["coupling_1d_to_2d_in"],
            "coupling_2d_to_1d_out_m3": mb2d["coupling_2d_to_1d_out"],
            "boundary_out_m3": mb2d["boundary_out"],
            "max_depth_2d_m": float(tri_max_depth.max()),
            "flood_volume_2d_m3": flood_volume_2d,
            "wet_area_2d_m2": float(tri_area[wet].sum()),
            "node_vol_flooded_total_m3": float(node_vol_flooded.sum()),
            "n_nodes_flooded": int((node_vol_flooded > 0).sum()),
            "node_max_depth_max_m": float(node_max_depth.max()),
        }

    arrays_path = inp_path.parent / "arrays.npz"
    np.savez_compressed(
        arrays_path,
        node_names=np.array(node_names), node_vol_flooded=node_vol_flooded,
        node_max_depth=node_max_depth, node_time_flooded=node_time_flooded,
        node_max_overflow=node_max_overflow,
        tri_max_depth=tri_max_depth, tri_max_vel=tri_max_vel, tri_area=tri_area,
    )
    return metrics


def _worker(args: tuple[str, str]) -> dict:
    inp_path, realization_id = args
    try:
        return run_one(Path(inp_path), realization_id)
    except Exception as e:  # noqa: BLE001 -- surface the failure, don't kill the pool
        return {"realization_id": realization_id, "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()}


def run_ensemble(
    base_sections, catalogue: pd.DataFrame, values: dict[str, pd.DataFrame],
    n_workers: int = config.N_WORKERS, force: bool = False,
    metrics_cache_path: "Path | None" = None,
) -> pd.DataFrame:
    """Write every realization's .inp (skipping ones already written unless
    `force`), run the ensemble in parallel, and return/append the cached
    metrics table.

    `metrics_cache_path` defaults to `config.METRICS_CACHE`; pass a distinct
    path when running a second, differently-shaped ensemble (e.g. the fine
    bearing sweep) so it doesn't overwrite the original ensemble's cache."""
    metrics_cache_path = metrics_cache_path or config.METRICS_CACHE
    from . import model as model_mod

    gage_ids = list(next(iter(values.values())).columns)
    jobs = []
    for _, row in catalogue.iterrows():
        rid = row.realization_id
        df = values[rid]
        run_dir = config.RUNS_DIR / rid
        inp_path = run_dir / f"{rid}.inp"
        arrays_path = run_dir / "arrays.npz"
        if not (inp_path.exists() and not force):
            inp_path = model_mod.write_realization_inp(
                base_sections, rid, gage_ids, df.index, df.values)
        if arrays_path.exists() and not force:
            continue
        jobs.append((str(inp_path), rid))

    results = []
    if metrics_cache_path.exists() and not force:
        results.append(pd.read_parquet(metrics_cache_path))

    if jobs:
        t0 = time.time()
        with Pool(n_workers) as pool:
            new_rows = pool.map(_worker, jobs)
        print(f"ran {len(jobs)} realization(s) in {time.time() - t0:.1f}s "
              f"({n_workers} workers)")
        failed = [r for r in new_rows if "error" in r]
        if failed:
            for r in failed:
                print(f"FAILED {r['realization_id']}: {r['error']}")
        results.append(pd.DataFrame([r for r in new_rows if "error" not in r]))

    metrics = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    if len(metrics):
        metrics = metrics.drop_duplicates(subset="realization_id", keep="last")
        catalogue_cols = ["realization_id", "bearing_deg", "direction_label", "flip",
                          "cross_shift_m", "domain_total_mm", "peak_centroid_x",
                          "peak_centroid_y", "event_centroid_x", "event_centroid_y"]
        # Idempotent: `metrics` may already carry these columns if it came
        # straight from the cached parquet (a prior run's merge). Drop
        # before re-merging so pandas never suffixes the collision as
        # `direction_label_x`/`_y`.
        metrics = metrics.drop(columns=[c for c in catalogue_cols if c != "realization_id"],
                               errors="ignore")
        metrics = metrics.merge(catalogue[catalogue_cols], on="realization_id", how="left")
        metrics.to_parquet(metrics_cache_path)
    return metrics


def load_arrays(realization_id: str) -> dict:
    d = np.load(config.RUNS_DIR / realization_id / "arrays.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def run_live_multi_engine(
    realization_ids: list[str], checkpoints: "list[timedelta] | None" = None,
    runs_dir: Path = config.RUNS_DIR,
) -> tuple[list[str], list[dict]]:
    """Hold every realization's `Solver` open *simultaneously* in this one
    process and interleave-step them to a shared sequence of elapsed-time
    checkpoints, capturing a live snapshot from every engine at each one.

    This is the actual reentrant-engine payoff: each `Solver` is an opaque,
    independent handle with no shared global state, so N of them can be
    live at once in a single Python process with no multiprocessing at
    all -- no OS-level process concurrency, so none of the segfault risk
    that came with running multiple engine *processes* in parallel earlier
    in this project. Every panel of the resulting animation is at the same
    simulated moment because they were advanced together, not independently
    completed runs replayed side by side.

    Returns (node_names, frames) where `frames` is one dict per checkpoint:
    {"elapsed": timedelta, "data": {realization_id: {"tri_depths": ndarray,
    "node_vol_flooded": ndarray}}}.
    """
    import sys
    if "rasterio" in sys.modules and "openswmm.engine" not in sys.modules:
        raise RuntimeError(
            "rasterio was imported before openswmm.engine in this process. "
            "Opening a second Solver while a first is still open segfaults "
            "in that import order (confirmed with a minimal repro) -- import "
            "openswmm.engine before `swmm_storm` (which pulls in rasterio via "
            "mesh.py) at the top of the notebook/script, then restart the "
            "kernel/process so the order actually takes effect."
        )

    from contextlib import ExitStack
    from datetime import timedelta
    from openswmm.engine import Solver, HAS_2D, Statistics

    if checkpoints is None:
        checkpoints = [timedelta(minutes=15 * k) for k in range(1, 25)]  # 15-min steps, 6h

    paths = {rid: runs_dir / rid / f"{rid}.inp" for rid in realization_ids}

    node_names: list[str] = []
    frames: list[dict] = []
    with ExitStack() as stack:
        solvers = {rid: stack.enter_context(Solver(str(p))) for rid, p in paths.items()}
        for s in solvers.values():
            if not (HAS_2D and s.surface2d.is_active):
                raise RuntimeError("2D module not active on one of the live-held solvers")
        node_names = [solvers[realization_ids[0]].nodes.get_id(i)
                     for i in range(len(Statistics(solvers[realization_ids[0]]).node_vol_flooded))]

        for target in checkpoints:
            frame = {"elapsed": target, "data": {}}
            for rid, s in solvers.items():
                s.until(target)
                surf = s.surface2d
                stats = Statistics(s)
                frame["data"][rid] = {
                    "tri_depths": surf.get_depths().copy(),
                    "node_vol_flooded": stats.node_vol_flooded.copy(),
                }
            frames.append(frame)

    return node_names, frames
