"""Shared paths and constants for the storm-trajectory demo.

All spatial work happens in EPSG:25832 (UTM32N, the Bellinge model's native
CRS) unless a variable name says otherwise (e.g. ``*_lonlat``).
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
#
# The Bellinge dataset is third-party published data and is not distributed
# with this notebook. Point BELLINGE_DATA_DIR at your own copy, or place it
# at <project>/data/Bellinge. See the README for where to obtain it.
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent

_env_dir = os.environ.get("BELLINGE_DATA_DIR")
BELLINGE_DIR = Path(_env_dir) if _env_dir else PROJECT_DIR / "data" / "Bellinge"

RADAR_DIR = BELLINGE_DIR / "Local_X-band"
INP_PATH = BELLINGE_DIR / "7_SWMM" / "BellingeSWMM_v021_nopervious.inp"
SUPPLIED_DEM_PATH = BELLINGE_DIR / "output_SRTMGL1.tif"

CACHE_DIR = PROJECT_DIR / "cache"
RUNS_DIR = CACHE_DIR / "runs"
RADAR_CUBE_CACHE = CACHE_DIR / "radar_cube.npz"
FOOTPRINT_CACHE = CACHE_DIR / "storm_footprint.npz"
REALIZATIONS_CACHE = CACHE_DIR / "realizations.parquet"
METRICS_CACHE = CACHE_DIR / "metrics.parquet"
DEM_CACHE = CACHE_DIR / "dem_25832.tif"
MESH_CACHE = CACHE_DIR / "mesh.npz"


def require_data() -> None:
    """Fail early and legibly if the Bellinge dataset is not where we expect.

    Without this the first failure is a FileNotFoundError several calls deep
    inside a radar parse or a raster read, which is a poor first experience
    for someone who has just cloned the repository.
    """
    missing = [p for p in (INP_PATH, RADAR_DIR, SUPPLIED_DEM_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Bellinge dataset not found. Expected these paths to exist:\n  "
            + "\n  ".join(str(p) for p in missing)
            + f"\n\nCurrently looking under: {BELLINGE_DIR}\n"
            "Set the BELLINGE_DATA_DIR environment variable to your copy of the\n"
            "dataset, or place it at <project>/data/Bellinge. The README explains\n"
            "where to download it."
        )

for d in (CACHE_DIR, RUNS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# CRS
# ---------------------------------------------------------------------------
CRS_MODEL = "EPSG:25832"   # Bellinge .inp coordinates (UTM32N)
CRS_RADAR = "EPSG:25832"   # radar pixel corners are already given in UTM32
CRS_WGS84 = "EPSG:4326"

# ---------------------------------------------------------------------------
# Radar grid geometry (from Local_X-band header coordinates; see radar.py)
# ---------------------------------------------------------------------------
RADAR_PIXEL_SIZE_M = 926.0
# Quarter-pixel bins gave the best validation R^2 in a 231/463/926/1389 m
# sweep against the observed single-cell passage (0.24 vs 0.15/0.15/0.16).
FOOTPRINT_BIN_SIZE_M = RADAR_PIXEL_SIZE_M / 4.0

# ---------------------------------------------------------------------------
# Storm event
# ---------------------------------------------------------------------------
EVENT_DATE = "2012-06-29"
EVENT_START = "2012-06-29 04:00:00"
EVENT_END = "2012-06-29 10:00:00"
# The 05:00-08:00 burst is genuinely multi-cellular: several discrete
# convective cells cross the domain west->east in succession, each resetting
# the intensity-weighted centroid as the previous one exits and a new one
# forms upstream. A straight-line fit over the whole burst averages this
# saw-tooth to near-zero velocity (speed ~0.1 m/s, R^2 ~0.01) -- not a motion
# estimate, just noise. See notebook Part 2 for the diagnostic.
#
# MOTION_FIT_WINDOW isolates the single most intense, cleanly-translating
# cell passage in the record (domain-mean peak 34 mm/hr at 06:18) and fits
# velocity to that alone: 6.8 m/s due east, R^2=0.87. FOOTPRINT_WINDOW pads
# this with lead-in/lead-out so the footprint captures the cell's full
# rise-decay shape without pulling in the unrelated cells before/after it.
MOTION_FIT_START = "2012-06-29 06:08:00"
MOTION_FIT_END = "2012-06-29 06:24:00"
FOOTPRINT_START = "2012-06-29 05:55:00"
FOOTPRINT_END = "2012-06-29 06:35:00"

# Kept for the multi-cell diagnostic plot (raw centroid trajectory only).
EVENT_BURST_START = "2012-06-29 05:00:00"
EVENT_BURST_END = "2012-06-29 08:00:00"

# QC: any pixel-day total above this (mm) is a bad radar record, not a storm
# (2017-08-20 reads ~2150 mm/day -- a known-bad file, excluded by this rule).
MAX_PLAUSIBLE_DAILY_MM = 200.0

# ---------------------------------------------------------------------------
# Realization ensemble (9-run default: 8 compass bearings + baseline)
# ---------------------------------------------------------------------------
# Meteorological "direction of travel" convention: bearing is where the storm
# is heading (not where it comes from), measured clockwise from north.
DIRECTION_LABELS = {
    0.0: "S->N",
    45.0: "SW->NE",
    90.0: "W->E",
    135.0: "NW->SE",
    180.0: "N->S",
    225.0: "NE->SW",
    270.0: "E->W",
    315.0: "SE->NW",
}

# ---------------------------------------------------------------------------
# Mesh / 2D
# ---------------------------------------------------------------------------
MESH_RES_M = 30.0          # triangle-grid cell size; coarsened if too slow
MESH_BUFFER_M = 250.0      # retain DEM cells within this distance of network
MANNINGS_N_2D = 0.05
COUPLING_CD = 0.65
COUPLING_AREA_M2 = 1.0

# ---------------------------------------------------------------------------
# Simulation window (trimmed from the .inp's full 2-day span for runtime)
# ---------------------------------------------------------------------------
SIM_START_DATE = "06/29/2012"
SIM_START_TIME = "04:00:00"
SIM_END_DATE = "06/29/2012"
SIM_END_TIME = "10:00:00"
REPORT_STEP = "00:05:00"

# Concurrent engine processes reproducibly SIGSEGV on this machine (crash
# reports land in ~/Library/Logs/DiagnosticReports at the exact moment),
# which silently hangs multiprocessing.Pool forever waiting on results that
# will never arrive -- a worker dying mid-task loses that task's Future with
# no exception raised anywhere. 4 workers crashed reliably; 2 workers ran
# clean once, then crashed on a later attempt -- so 2 is NOT safe, only
# less likely to fail. A single engine instance (no concurrency) has never
# crashed. Serial is the only currently-verified-safe setting; only raise
# this once the underlying native race is root-caused (likely shared-state
# initialization in a dependency, given each process's own engine handle is
# supposed to be independent per the reentrant design).
N_WORKERS = 1
FLOOD_DEPTH_THRESHOLD_M = 0.05  # "wet" cell threshold for spatial box plots
