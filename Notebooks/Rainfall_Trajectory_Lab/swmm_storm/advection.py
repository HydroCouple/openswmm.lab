"""Frozen-field storm resynthesis.

The Local X-band radar grid is only 9x7 pixels (926 m spacing) -- too coarse
for image cross-correlation to resolve a sub-pixel displacement between
consecutive frames reliably. Instead we track the intensity-weighted
rainfall **centroid** through the burst window and fit its trajectory with a
straight line; this is the same principle storm-cell tracking algorithms
(e.g. TITAN) use, and it is well matched to a coarse grid because it uses
every pixel's value at every timestep rather than a handful of displaced
features.

Given that motion estimate, every observed (x, y, t, rain-rate) sample is
re-expressed in a storm-centered, along-/cross-track frame -- the "footprint".
If Taylor's frozen-turbulence hypothesis holds over the burst, that footprint
is close to time-invariant, and any new bearing/speed/offset can be used to
re-project it back into real space to synthesize a trajectory the radar never
observed, without ever clipping the storm's edges.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import RegularGridInterpolator

from . import config


# ---------------------------------------------------------------------------
# Bearing <-> vector helpers (compass convention: 0 = N, 90 = E, clockwise)
# ---------------------------------------------------------------------------
def bearing_to_unit(bearing_deg: float) -> np.ndarray:
    theta = np.radians(bearing_deg)
    return np.array([np.sin(theta), np.cos(theta)])  # (east, north) components


def vector_to_bearing(vx: float, vy: float) -> float:
    return float(np.degrees(np.arctan2(vx, vy)) % 360.0)


def _cross_of(along_hat: np.ndarray) -> np.ndarray:
    """Unit vector 90 deg counter-clockwise from along_hat."""
    return np.array([-along_hat[1], along_hat[0]])


# ---------------------------------------------------------------------------
# Step 1 -- centroid trajectory and motion fit
# ---------------------------------------------------------------------------
def centroid_trajectory(cube: dict) -> pd.DataFrame:
    """Intensity-weighted rainfall centroid at every cube timestep."""
    R, x, y, times = cube["R"], cube["x"], cube["y"], cube["times"]
    xx, yy = np.meshgrid(x, y)  # shape (n_rows, n_cols), matches R[:, r, c]
    total = R.sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        cx = (R * xx).sum(axis=(1, 2)) / total
        cy = (R * yy).sum(axis=(1, 2)) / total
    return pd.DataFrame({"time": times, "total_mm_hr": total, "cx": cx, "cy": cy})


@dataclass
class MotionEstimate:
    vx: float           # m/s, east component
    vy: float           # m/s, north component
    speed_ms: float
    bearing_deg: float  # observed direction of travel
    t0: pd.Timestamp    # reference time (peak domain-mean intensity)
    c0: np.ndarray       # centroid (x, y) at t0, from the fitted line
    r2_x: float
    r2_y: float
    fit_df: pd.DataFrame = field(repr=False)


def estimate_motion(
    cube: dict,
    burst_start: str = config.MOTION_FIT_START,
    burst_end: str = config.MOTION_FIT_END,
    min_activity_frac: float = 0.05,
) -> MotionEstimate:
    """Fit a straight-line velocity to the centroid trajectory over `burst`.

    Only timesteps with domain-mean intensity above `min_activity_frac` of
    the window's peak are used -- the centroid of a nearly-dry frame is
    numerically unstable and would otherwise dominate the noise.

    Defaults to `config.MOTION_FIT_START/END`, a ~16-minute window isolating
    the event's single most intense, coherently-translating convective cell.
    Fitting the same straight line over the full 3-hour burst instead
    (`config.EVENT_BURST_START/END`) collapses to near-zero speed and R^2 --
    the burst is multi-cellular, not one coherent moving body. See
    `centroid_trajectory` for the raw diagnostic.
    """
    traj = centroid_trajectory(cube)
    burst = traj[(traj.time >= burst_start) & (traj.time < burst_end)].copy()
    threshold = min_activity_frac * burst.total_mm_hr.max()
    active = burst[burst.total_mm_hr >= threshold].copy()
    if len(active) < 3:
        raise ValueError(
            "Too few active radar frames in the burst window to fit a storm "
            "motion vector -- widen EVENT_BURST_START/END or lower min_activity_frac."
        )

    t0 = burst.loc[burst.total_mm_hr.idxmax(), "time"]
    t_sec = (active.time - t0).dt.total_seconds().values

    fit_x = stats.linregress(t_sec, active.cx.values)
    fit_y = stats.linregress(t_sec, active.cy.values)
    vx, vy = fit_x.slope, fit_y.slope
    c0 = np.array([fit_x.intercept, fit_y.intercept])  # position at t0 (t_sec=0)

    speed = float(np.hypot(vx, vy))
    bearing = vector_to_bearing(vx, vy)

    active = active.assign(t_sec=t_sec,
                            cx_fit=fit_x.intercept + fit_x.slope * t_sec,
                            cy_fit=fit_y.intercept + fit_y.slope * t_sec)
    return MotionEstimate(
        vx=vx, vy=vy, speed_ms=speed, bearing_deg=bearing, t0=t0, c0=c0,
        r2_x=fit_x.rvalue ** 2, r2_y=fit_y.rvalue ** 2, fit_df=active,
    )


# ---------------------------------------------------------------------------
# Step 2 -- unwrap into a storm-centered footprint
# ---------------------------------------------------------------------------
@dataclass
class Footprint:
    u_centers: np.ndarray   # along-track axis, m, storm-relative
    w_centers: np.ndarray   # cross-track axis, m, storm-relative
    F: np.ndarray           # (n_w, n_u) mean mm/hr per bin
    counts: np.ndarray      # samples per bin (for QC)
    bin_size: float


def _slice_window(cube: dict, window: tuple[str, str] | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    R, x, y, times = cube["R"], cube["x"], cube["y"], cube["times"]
    if window is None:
        return R, x, y, times
    start, end = window
    mask = (times >= start) & (times < end)
    return R[mask], x, y, times[mask]


def build_footprint(
    cube: dict, motion: MotionEstimate, bin_size: float | None = None,
    window: tuple[str, str] | None = (config.FOOTPRINT_START, config.FOOTPRINT_END),
) -> Footprint:
    """Unwrap radar samples in `window` (default: the padded single-cell
    passage, see `config.FOOTPRINT_START/END`) into a storm-centered grid.
    Pass `window=None` to use every frame in `cube` instead."""
    R, x, y, times = _slice_window(cube, window)
    bin_size = bin_size or config.FOOTPRINT_BIN_SIZE_M
    along_hat = np.array([motion.vx, motion.vy]) / motion.speed_ms
    cross_hat = _cross_of(along_hat)

    xx, yy = np.meshgrid(x, y)  # (n_rows, n_cols)
    t_sec = (times - motion.t0).total_seconds().values  # (T,)

    # Broadcast: storm-relative position of every (t, row, col) sample.
    cx_t = motion.c0[0] + motion.vx * t_sec  # (T,)
    cy_t = motion.c0[1] + motion.vy * t_sec
    rel_x = xx[None, :, :] - cx_t[:, None, None]
    rel_y = yy[None, :, :] - cy_t[:, None, None]
    u = rel_x * along_hat[0] + rel_y * along_hat[1]
    w = rel_x * cross_hat[0] + rel_y * cross_hat[1]

    u_flat, w_flat, v_flat = u.ravel(), w.ravel(), R.ravel()
    u_edges = np.arange(u_flat.min() - bin_size, u_flat.max() + 2 * bin_size, bin_size)
    w_edges = np.arange(w_flat.min() - bin_size, w_flat.max() + 2 * bin_size, bin_size)

    sums, _, _ = np.histogram2d(w_flat, u_flat, bins=[w_edges, u_edges], weights=v_flat)
    counts, _, _ = np.histogram2d(w_flat, u_flat, bins=[w_edges, u_edges])
    with np.errstate(invalid="ignore", divide="ignore"):
        F = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)

    u_centers = 0.5 * (u_edges[:-1] + u_edges[1:])
    w_centers = 0.5 * (w_edges[:-1] + w_edges[1:])
    return Footprint(u_centers=u_centers, w_centers=w_centers, F=F, counts=counts, bin_size=bin_size)


def _footprint_interpolator(fp: Footprint) -> RegularGridInterpolator:
    return RegularGridInterpolator(
        (fp.w_centers, fp.u_centers), fp.F,
        bounds_error=False, fill_value=0.0, method="linear",
    )


# ---------------------------------------------------------------------------
# Step 3 -- validate: re-advect at the OBSERVED bearing/speed and compare
# ---------------------------------------------------------------------------
def validate_footprint(
    cube: dict, motion: MotionEstimate, fp: Footprint,
    window: tuple[str, str] | None = (config.FOOTPRINT_START, config.FOOTPRINT_END),
) -> pd.DataFrame:
    """Reconstruct each pixel's series from the footprint at the observed
    motion and compare to what the radar actually recorded over `window`
    (default: the same padded single-cell window the footprint was built
    from). Returns one row per pixel with R^2 and bias (mm/hr).
    """
    R, x, y, times = _slice_window(cube, window)
    interp = _footprint_interpolator(fp)
    along_hat = np.array([motion.vx, motion.vy]) / motion.speed_ms
    cross_hat = _cross_of(along_hat)
    t_sec = (times - motion.t0).total_seconds().values
    cx_t = motion.c0[0] + motion.vx * t_sec
    cy_t = motion.c0[1] + motion.vy * t_sec

    rows = []
    n_rows, n_cols = R.shape[1], R.shape[2]
    for r in range(n_rows):
        for c in range(n_cols):
            rel_x = x[c] - cx_t
            rel_y = y[r] - cy_t
            u = rel_x * along_hat[0] + rel_y * along_hat[1]
            w = rel_x * cross_hat[0] + rel_y * cross_hat[1]
            pred = interp(np.column_stack([w, u]))
            obs = R[:, r, c]
            ss_res = np.sum((obs - pred) ** 2)
            ss_tot = np.sum((obs - obs.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            rows.append({"row": r, "col": c, "x": x[c], "y": y[r],
                         "r2": r2, "bias_mm_hr": float(np.mean(pred - obs)),
                         "obs_mean": float(obs.mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 4/5 -- realization catalogue, synthesis, and renormalization
# ---------------------------------------------------------------------------
def compass_bearings(n: int) -> dict[float, str]:
    """`n` bearings evenly spaced around the compass (360/n degrees apart),
    labeled by degree (e.g. "015") rather than a named compass point -- at
    fine spacing (n > 8) most bearings don't land on a named 8/16-point
    direction, and a numeric label stays unambiguous at any resolution.
    Plain digits only (no degree symbol) since this label also becomes part
    of a filesystem path via `realization_spec`."""
    step = 360.0 / n
    return {i * step: f"{i * step:03.0f}" for i in range(n)}


def realization_spec(
    bearings: dict[float, str] = config.DIRECTION_LABELS,
    flips: tuple[bool, ...] = (False,),
    cross_shifts_m: tuple[float, ...] = (0.0,),
    include_baseline: bool = True,
    id_prefix: str = "R",
) -> pd.DataFrame:
    """`id_prefix` namespaces `realization_id` (and hence each run's cache
    directory under `config.RUNS_DIR`) so a second, differently-shaped
    ensemble -- e.g. a fine bearing sweep for an animation -- can coexist
    with the original one without colliding IDs or overwriting its runs."""
    rows = []
    if include_baseline:
        rows.append({"realization_id": f"{id_prefix}00_baseline", "bearing_deg": None,
                     "direction_label": "observed", "flip": False, "cross_shift_m": 0.0})
    i = 1
    for bearing, label in sorted(bearings.items()):
        for flip in flips:
            for shift in cross_shifts_m:
                suffix = f"_flip" if flip else ""
                suffix += f"_s{shift:+.0f}" if shift else ""
                rid = f"{id_prefix}{i:02d}_{label.replace('->', '-')}{suffix}"
                rows.append({"realization_id": rid, "bearing_deg": bearing,
                             "direction_label": label, "flip": flip,
                             "cross_shift_m": shift})
                i += 1
    return pd.DataFrame(rows)


def synthesize_series(
    motion: MotionEstimate, fp: Footprint,
    query_xy: np.ndarray, query_times: pd.DatetimeIndex,
    bearing_deg: float | None, flip: bool, cross_shift_m: float,
) -> np.ndarray:
    """Sample the footprint along a (possibly new) trajectory.

    `bearing_deg=None` reproduces the observed trajectory (the baseline
    realization). Speed is always held at the observed magnitude; only
    direction, mirroring, and cross-track offset change.

    Returns an (T, N) array of mm/hr, T = len(query_times), N = len(query_xy).
    """
    if bearing_deg is None:
        along_hat = np.array([motion.vx, motion.vy]) / motion.speed_ms
    else:
        along_hat = bearing_to_unit(bearing_deg)
    cross_hat = _cross_of(along_hat)
    v_new = along_hat * motion.speed_ms

    t_sec = (query_times - motion.t0).total_seconds().values
    c0 = motion.c0 + cross_shift_m * cross_hat
    cx_t = c0[0] + v_new[0] * t_sec
    cy_t = c0[1] + v_new[1] * t_sec

    interp = _footprint_interpolator(fp)
    out = np.zeros((len(query_times), len(query_xy)))
    for j, (qx, qy) in enumerate(query_xy):
        rel_x = qx - cx_t
        rel_y = qy - cy_t
        u = rel_x * along_hat[0] + rel_y * along_hat[1]
        w = rel_x * cross_hat[0] + rel_y * cross_hat[1]
        if flip:
            w = -w
        out[:, j] = interp(np.column_stack([w, u]))
    return np.maximum(out, 0.0)


def domain_total_mm(values_mm_hr: np.ndarray, dt_minutes: float = 1.0) -> float:
    """Catchment-average rainfall depth (mm) implied by a (T, N) mm/hr array,
    averaging over query points (N) and integrating over time (T)."""
    return float(values_mm_hr.mean(axis=1).sum() * dt_minutes / 60.0)


def compute_centroids(
    query_xy: np.ndarray, query_times: pd.DatetimeIndex, values_mm_hr: np.ndarray,
) -> dict:
    """Peak-timestep and event-total intensity-weighted centroids, in the
    same (x, y) space as query_xy (EPSG:25832 m)."""
    domain_mean = values_mm_hr.mean(axis=1)
    peak_idx = int(np.argmax(domain_mean))
    peak_w = values_mm_hr[peak_idx]
    peak_total = peak_w.sum()
    if peak_total > 0:
        peak_cx = float((peak_w * query_xy[:, 0]).sum() / peak_total)
        peak_cy = float((peak_w * query_xy[:, 1]).sum() / peak_total)
    else:
        peak_cx = peak_cy = float("nan")

    event_w = values_mm_hr.sum(axis=0)  # proportional to depth per point
    event_total = event_w.sum()
    if event_total > 0:
        event_cx = float((event_w * query_xy[:, 0]).sum() / event_total)
        event_cy = float((event_w * query_xy[:, 1]).sum() / event_total)
    else:
        event_cx = event_cy = float("nan")

    return {
        "peak_time": query_times[peak_idx],
        "peak_centroid_x": peak_cx, "peak_centroid_y": peak_cy,
        "event_centroid_x": event_cx, "event_centroid_y": event_cy,
    }


def build_realizations(
    cube: dict, motion: MotionEstimate, fp: Footprint,
    gage_xy: np.ndarray, gage_ids: list[str],
    query_times: pd.DatetimeIndex | None = None,
    bearings: dict[float, str] = config.DIRECTION_LABELS,
    flips: tuple[bool, ...] = (False,),
    cross_shifts_m: tuple[float, ...] = (0.0,),
    force: bool = False,
    id_prefix: str = "R",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Synthesize every realization's gage time series and its provenance
    row, renormalizing every non-baseline realization to the baseline's
    domain-total rainfall so trajectory is the only thing that varies.

    `id_prefix` (see `realization_spec`) lets a second, differently-shaped
    ensemble coexist with a prior one under distinct realization IDs / cache
    directories.

    Returns (catalogue_df, {realization_id: DataFrame[time x gage_ids]}).
    """
    query_times = query_times if query_times is not None else cube["times"]
    spec = realization_spec(bearings, flips, cross_shifts_m, id_prefix=id_prefix)

    values: dict[str, pd.DataFrame] = {}
    baseline_total = None
    rows = []
    for _, row in spec.iterrows():
        # NB: the baseline's bearing_deg is Python None in realization_spec,
        # but once that column sits in a DataFrame next to float bearings,
        # pandas silently upcasts None -> NaN. `pd.isna` catches both.
        bearing = None if pd.isna(row.bearing_deg) else row.bearing_deg
        raw = synthesize_series(
            motion, fp, gage_xy, query_times,
            bearing_deg=bearing, flip=row.flip, cross_shift_m=row.cross_shift_m,
        )
        total = domain_total_mm(raw)
        if row.direction_label == "observed":
            baseline_total = total
            renorm_factor = 1.0
            scaled = raw
        else:
            renorm_factor = baseline_total / total if total > 0 else float("nan")
            scaled = raw * renorm_factor

        cen = compute_centroids(gage_xy, query_times, scaled)
        df = pd.DataFrame(scaled, index=query_times, columns=gage_ids)
        values[row.realization_id] = df

        rows.append({
            **row.to_dict(),
            "speed_ms": motion.speed_ms,
            "domain_total_mm": domain_total_mm(scaled),
            "renorm_factor": renorm_factor,
            **cen,
        })

    catalogue = pd.DataFrame(rows)
    return catalogue, values
