"""Parse the Bellinge Local X-band radar archive into a space-time cube.

``Local_X-band/`` holds one file per radar pixel: 58 files on a nominal
9x7 grid at 926 m spacing (5 of 63 cells are absent). Each file's header
gives the pixel's lower-left UTM32 corner; the body is
``DateTime [UTC];value[um/s]`` at (mostly) 1-minute cadence, with gaps during
dry periods.

Values are micrometers/second; multiplying by 3.6 gives mm/hr.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

_HEADER_RE = re.compile(
    r"Id:(?P<id>\d+)LowerLeft-UTM32:\((?P<x>-?\d+),\s*(?P<y>-?\d+)\)"
)

UM_S_TO_MM_HR = 3.6


@dataclass(frozen=True)
class PixelInfo:
    pixel_id: int
    path: Path
    x0: float  # lower-left corner, m (UTM32)
    y0: float
    col: int   # grid column index (0-based, west->east)
    row: int   # grid row index (0-based, south->north)

    @property
    def x_center(self) -> float:
        return self.x0 + config.RADAR_PIXEL_SIZE_M / 2.0

    @property
    def y_center(self) -> float:
        return self.y0 + config.RADAR_PIXEL_SIZE_M / 2.0


def _parse_header(path: Path) -> tuple[int, float, float]:
    with open(path, "r", errors="ignore") as f:
        f.readline()  # "Data source : ..."
        header = f.readline()
    m = _HEADER_RE.search(header)
    if not m:
        raise ValueError(f"Unrecognised radar header in {path}: {header!r}")
    return int(m["id"]), float(m["x"]), float(m["y"])


def discover_pixels(radar_dir: Path = config.RADAR_DIR) -> list[PixelInfo]:
    """Scan every radar file's header and lay pixels out on a regular grid.

    Grid indices are derived from the *set* of distinct corner coordinates
    actually present, not an assumed full 9x7 rectangle -- this is what lets
    the 5 missing cells fall out naturally as grid holes.
    """
    files = sorted(radar_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No radar files under {radar_dir}")

    raw = [(*_parse_header(p), p) for p in files]
    xs = sorted({x for _, x, _, _ in raw})
    ys = sorted({y for _, _, y, _ in raw})
    col_of = {x: i for i, x in enumerate(xs)}
    row_of = {y: i for i, y in enumerate(ys)}

    pixels = [
        PixelInfo(pixel_id=pid, path=p, x0=x, y0=y, col=col_of[x], row=row_of[y])
        for pid, x, y, p in raw
    ]
    return pixels


def _load_series(path: Path, start: str, end: str) -> pd.Series:
    """Load one pixel's mm/hr series over [start, end), reindexed to 1-min."""
    df = pd.read_csv(
        path, sep=";", skiprows=3, header=None, names=["time", "value"],
        engine="c", parse_dates=["time"],
    )
    df = df[(df.time >= start) & (df.time < end)]
    s = df.set_index("time")["value"].sort_index()
    s = s[~s.index.duplicated(keep="last")] * UM_S_TO_MM_HR
    full_index = pd.date_range(start, end, freq="1min", inclusive="left")
    # Missing minutes inside a captured event window are dry gaps, not
    # missing instrument data -- the archive only writes non-trivial spans
    # densely; fill with 0 mm/hr.
    return s.reindex(full_index, fill_value=0.0)


def _idw_fill(grid: np.ndarray, present_rc: list[tuple[int, int]],
              missing_rc: list[tuple[int, int]], power: float = 2.0) -> np.ndarray:
    """Fill missing (row, col) grid cells via inverse-distance weighting
    from present cells. `grid` has shape (T, n_rows, n_cols) or (n_rows, n_cols).
    """
    out = grid.copy()
    present_arr = np.array(present_rc, dtype=float)
    for r, c in missing_rc:
        d = np.hypot(present_arr[:, 0] - r, present_arr[:, 1] - c)
        w = 1.0 / np.power(np.maximum(d, 1e-9), power)
        w /= w.sum()
        if grid.ndim == 3:
            acc = np.zeros(grid.shape[0])
            for (pr, pc), wi in zip(present_rc, w):
                acc += wi * grid[:, pr, pc]
            out[:, r, c] = acc
        else:
            out[r, c] = sum(wi * grid[pr, pc] for (pr, pc), wi in zip(present_rc, w))
    return out


def build_event_cube(
    start: str = config.EVENT_START,
    end: str = config.EVENT_END,
    force: bool = False,
) -> dict:
    """Build (and cache) the (T, n_rows, n_cols) mm/hr radar cube for [start, end).

    Returns a dict: times, R (mm/hr), x (col centers, m), y (row centers, m),
    pixel_id_grid (-1 = interpolated), filled_mask, qc (per-pixel DataFrame).
    """
    if config.RADAR_CUBE_CACHE.exists() and not force:
        d = np.load(config.RADAR_CUBE_CACHE, allow_pickle=True)
        if str(d["start"]) == start and str(d["end"]) == end:
            return {
                "times": pd.to_datetime(d["times"]),
                "R": d["R"],
                "x": d["x"],
                "y": d["y"],
                "pixel_id_grid": d["pixel_id_grid"],
                "filled_mask": d["filled_mask"],
                "qc": pd.DataFrame(d["qc"].item()),
            }

    pixels = discover_pixels()
    n_rows = max(p.row for p in pixels) + 1
    n_cols = max(p.col for p in pixels) + 1

    full_index = pd.date_range(start, end, freq="1min", inclusive="left")
    T = len(full_index)
    R = np.full((T, n_rows, n_cols), np.nan, dtype=np.float64)
    pixel_id_grid = np.full((n_rows, n_cols), -1, dtype=np.int64)

    qc_rows = []
    for p in pixels:
        s = _load_series(p.path, start, end)
        R[:, p.row, p.col] = s.values
        pixel_id_grid[p.row, p.col] = p.pixel_id
        qc_rows.append({
            "pixel_id": p.pixel_id, "row": p.row, "col": p.col,
            "x": p.x_center, "y": p.y_center,
            "event_total_mm": s.sum() / 60.0,  # mm/hr * (1/60 hr) per minute
            "event_peak_mm_hr": s.max(),
        })
    qc = pd.DataFrame(qc_rows)

    bad = qc[qc.event_total_mm > config.MAX_PLAUSIBLE_DAILY_MM]
    if len(bad):
        raise ValueError(
            f"Radar QC failed for window {start}..{end}: implausible totals "
            f"at pixels {bad.pixel_id.tolist()} (> {config.MAX_PLAUSIBLE_DAILY_MM} mm). "
            "This is the corrupt-file failure mode seen on 2017-08-20 -- check the "
            "event window."
        )

    present_rc = [(p.row, p.col) for p in pixels]
    all_rc = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    missing_rc = [rc for rc in all_rc if rc not in present_rc]
    filled_mask = np.zeros((n_rows, n_cols), dtype=bool)
    for r, c in missing_rc:
        filled_mask[r, c] = True

    if missing_rc:
        R = _idw_fill(R, present_rc, missing_rc)

    # pixels with the same col share x_center; same row share y_center
    x = np.zeros(n_cols)
    y = np.zeros(n_rows)
    for p in pixels:
        x[p.col] = p.x_center
        y[p.row] = p.y_center

    result = {
        "times": full_index, "R": R, "x": x, "y": y,
        "pixel_id_grid": pixel_id_grid, "filled_mask": filled_mask, "qc": qc,
    }
    np.savez_compressed(
        config.RADAR_CUBE_CACHE,
        start=start, end=end, times=full_index.values, R=R, x=x, y=y,
        pixel_id_grid=pixel_id_grid, filled_mask=filled_mask,
        qc=qc.to_dict(orient="list"),
    )
    return result


def rank_event_dates(pixel_id: int = 895, n: int = 15) -> pd.Series:
    """Daily rainfall totals (mm) at one pixel, full record, ranked descending.

    Used to document event selection (2012-06-29 chosen; 2017-08-20 excluded
    as a corrupt record) without re-parsing all 58 files' full history.
    """
    pixels = {p.pixel_id: p for p in discover_pixels()}
    if pixel_id not in pixels:
        pixel_id = next(iter(pixels))
    df = pd.read_csv(
        pixels[pixel_id].path, sep=";", skiprows=3, header=None,
        names=["time", "value"], engine="c", parse_dates=["time"],
    )
    df["date"] = df["time"].dt.date
    daily_mm = df.groupby("date")["value"].sum() * (UM_S_TO_MM_HR / 60.0)
    return daily_mm.sort_values(ascending=False).head(n)


def domain_mean_series(cube: dict) -> pd.Series:
    """Domain-mean rainfall intensity (mm/hr) time series for a cube."""
    return pd.Series(cube["R"].mean(axis=(1, 2)), index=cube["times"])
