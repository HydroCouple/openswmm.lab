"""Generates notebooks/bellinge_storm_trajectories.ipynb from cell source
strings below. Run once (and re-run after editing this file) rather than
hand-editing notebook JSON.
"""
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
# Local layout keeps the notebook under notebooks/; the published lab folder
# keeps it at the root. Support both without duplicating the file.
NB_PATH = (_ROOT / "notebooks" if (_ROOT / "notebooks").is_dir() else _ROOT) / "bellinge_storm_trajectories.ipynb"


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


cells = []

# ---------------------------------------------------------------------------
cells.append(md("""\
# Storm Trajectory Sensitivity on the Bellinge 1D/2D Model

Two storms drop exactly the same amount of water on the same city. The first sweeps in from the west and tracks
east. The second climbs up from the south. Do they flood the same streets, to the same depth?

Standard practice mostly assumes they do. A design storm is a rainfall depth and a duration applied everywhere at
once, so it has no direction to speak of. This notebook tests that assumption directly. We take one real convective
storm recorded over the Bellinge catchment in Odense, Denmark, make nine copies of it traveling along nine different
compass bearings, force every copy to deliver an identical total volume of water, and run each through a coupled
pipe and surface model of the real drainage network. Whatever differences appear at the end can only have come from
the direction of travel, because that is the only thing we allowed to vary.

## What a reentrant engine has to do with any of this

Picture a chess grandmaster playing a simultaneous exhibition. Twenty boards are set up in a row, each holding its
own position, and the grandmaster walks down the line playing one move per board. Every game advances together.

Many simulation engines work the opposite way. They keep the state of the running simulation in global memory, so
there is only ever one board in the room. You have to finish a game before you can set up the next one.

`openswmm.engine` is reentrant, which means each `Solver` object carries its entire simulation inside itself and
shares nothing outside. That gives us the row of boards. Most of this notebook does not need it, since running an
ensemble one simulation at a time works perfectly well. Part 7 is where it earns its place, when six storms are
opened at once and stepped forward together so that every frame of an animation shows six possible futures of the
same catchment at the same instant in simulated time.

## How the notebook is laid out

We start with the drainage network itself, then pull a real storm out of a radar archive. We measure how fast and in
which direction that storm was moving, which lets us rebuild it as a portable object we can send across the
catchment on any heading we choose. Then we drape a triangular surface over the terrain and connect it to the pipes,
test a single run to confirm the physics is behaving, and finally launch the full comparison.
"""))

# ---------------------------------------------------------------------------
cells.append(code("""\
import os
# Must be set before `openswmm.engine` is imported anywhere in this process:
# the engine links Homebrew's libomp while the conda numpy/scipy stack loads
# its own copy, and macOS aborts on the duplicate OpenMP runtime otherwise.
# This is the standard, documented workaround for that conda + Homebrew
# conflict -- not a numerical hazard.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import sys
from pathlib import Path
# repo root = parent of notebooks/, so `swmm_storm` is importable
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))

# Import order matters here: `openswmm.engine` must load before `rasterio`
# (pulled in transitively by `swmm_storm.mesh`) is imported anywhere in this
# process. Both bundle their own copies of shared native dependencies
# (GDAL's stack vs. openswmm's vcpkg-built libraries), and importing
# rasterio first causes a segfault the moment a SECOND `Solver` is opened
# while a first one is still open -- reproduced with a 2-line repro; a
# single Solver open/close at a time is unaffected either way, which is why
# this never surfaced in the ensemble runs earlier in the notebook.
import openswmm.engine  # noqa: F401 -- import-order guard, see above

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from swmm_storm import config, inp_utils as iu, radar, advection as adv, mesh as mesh_mod, model, run, figures

# Set to True to force every cached stage (radar cube, mesh, ensemble runs) to
# recompute from scratch. Leave False for a fast, cache-backed re-run.
FORCE_RERUN = False

# Fail early with a readable message if the Bellinge dataset is not in place.
# It is third-party data and is not shipped with this repository; see README.
config.require_data()

pd.set_option("display.width", 160)
plt.rcParams["figure.facecolor"] = "white"
print("openswmm.engine HAS_2D:", __import__("openswmm.engine", fromlist=["HAS_2D"]).HAS_2D)
"""))

# ---------------------------------------------------------------------------
cells.append(md("""\
## Part 1. The drainage network we are testing

Bellinge is a neighborhood in Odense, Denmark, and its sewer system is one of the few real networks published openly
enough to experiment with. The model contains 995 junctions, which you can picture as manholes, along with 16
storage structures such as detention basins, and 713 subcatchments, which are the parcels of land that shed
rainwater into each pipe.

The published version is driven by two rain gages covering the whole catchment. That is completely normal practice,
and it is also exactly the limitation we want to remove, because two gages cannot tell you which end of town got hit
first. Over the next few sections we replace them with 58 radar pixels and add a ground surface so that water has
somewhere to go when a pipe fills past capacity. The pipe network itself stays exactly as published.
"""))

cells.append(code("""\
inp_text = open(config.INP_PATH, errors="ignore").read()
sections = iu.split_sections(inp_text)
nodes = iu.read_node_table(sections)
poly_xy = iu.read_polygon_vertices(sections)
outlets = iu.read_subcatchment_outlets(sections)

print(f"{len(nodes)} nodes ({(nodes.kind=='JUNCTIONS').sum()} junctions, {(nodes.kind=='STORAGE').sum()} storage)")
print(f"{len(outlets)} subcatchments, {len(poly_xy)} polygon vertices")
print(f"network extent (EPSG:25832): x [{nodes.x.min():.0f}, {nodes.x.max():.0f}]  "
      f"y [{nodes.y.min():.0f}, {nodes.y.max():.0f}]")
print(f"rim elevation range: {nodes.rim.min():.1f} - {nodes.rim.max():.1f} m")

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(poly_xy[:, 0], poly_xy[:, 1], s=1, color="#d8d7d2", label="subcatchments")
ax.scatter(nodes.x, nodes.y, s=4, color="#2a78d6", label="nodes")
ax.set_aspect("equal"); ax.set_title("Bellinge network", loc="left")
ax.legend(frameon=False, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout()
"""))

# ---------------------------------------------------------------------------
cells.append(md("""\
## Part 2. Finding a real storm worth moving around

A rain gage is a microphone. It tells you loudness over time at a single spot. Weather radar is closer to a video
camera, recording the whole scene at once, and that difference is what makes this notebook possible at all. You
cannot move a storm you only measured at two points, because you never saw its shape.

The Bellinge archive holds X-band radar from 2012 through 2020 as 58 separate files, one per pixel, on a grid of
926 meter squares sampled every minute. Stacked together they form a rainfall movie.

Before trusting any of it, we rank the wettest days, and the top result turns out to be a warning rather than a
storm. One pixel reports roughly 2150 mm in a single day, about a hundred times a heavy Danish downpour and
comfortably beyond the world record. That is a sensor fault, and a quality check screens it out. Skipping this step
would mean modeling a catastrophe that never happened, and every flood number after it would be fiction.

The genuine record holder is 29 June 2012. Between 05:00 and 08:00 it delivered 20 to 43 mm across the catchment,
with peak intensities of 93 to 186 mm/hr. It is fast, compact and convective, which is the type of storm where
direction of travel should matter most. A slow, wide frontal system soaks everything fairly evenly, so its heading
changes little. A tight, intense cell behaves more like a moving spotlight, and where it points is the whole
question.
"""))

cells.append(code("""\
ranked = radar.rank_event_dates(pixel_id=895, n=8)
display(ranked.to_frame("total_mm"))
"""))

cells.append(code("""\
cube = radar.build_event_cube(force=FORCE_RERUN)
print(f"cube: {cube['R'].shape} (time, row, col), {cube['filled_mask'].sum()} interpolated cells")
display(cube["qc"].sort_values("event_total_mm", ascending=False).head())

domain_mean = radar.domain_mean_series(cube)
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.plot(domain_mean.index, domain_mean.values, color="#2a78d6", linewidth=1.4)
ax.set_ylabel("domain-mean rain rate (mm/hr)"); ax.set_title(
    "2012-06-29 event: observed domain-mean intensity", loc="left")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout()
"""))

# ---------------------------------------------------------------------------
cells.append(md("""\
## Part 3. Turning one storm into nine

Imagine a patterned rug being dragged across a floor. Stand still, watch one spot, and the pattern beneath you keeps
changing. The rug itself is not changing at all. It is simply moving. If you record everything that passes that one
spot, and you know how fast the rug is sliding and in which direction, you can reconstruct the whole pattern. Once
you hold the pattern, you can drag it across the floor any way you please.

That idea is Taylor's frozen field hypothesis, and it is how we get nine storms out of one. We track the center of
mass of the rainfall as it crosses the catchment, measure its velocity, then convert every radar reading from "what
fell at this place at this time" into "what fell at this position relative to the storm's own center." The result is
a portable map of the storm, which we call its footprint. Sending it back across the catchment on a new heading
becomes arithmetic.

Working this way rather than rotating the raw pixel grid matters practically. The observed grid is only nine squares
by seven, so turning it 45 degrees would slice the corners off the storm and quietly delete rainfall.

The measurement has an honest complication that is worth showing rather than hiding. Fitting a single velocity to
the entire three hour burst returns almost zero speed and an R squared near 0.01, which is a failed fit rather than
a slow storm. The plot below explains why. The burst is a series of separate convective cells forming upstream,
crossing, and dissipating, so the center of mass jumps backward every time one cell hands off to the next. We
therefore isolate the single cleanest cell passage, about sixteen minutes around the peak, and fit that.

The cost of this approach is worth stating plainly. The footprint reproduces roughly a quarter of the variance at
any individual pixel, because real cells grow, rotate and decay while they travel, and a frozen field cannot
represent any of that. The storm's overall movement is captured well and its fine internal detail is not. That
tradeoff is acceptable here because the question is where the core of the storm travels. It would not be acceptable
if you needed to reproduce a specific gage trace faithfully.
"""))

cells.append(code("""\
traj = adv.centroid_trajectory(cube)
burst = traj[(traj.time >= config.EVENT_BURST_START) & (traj.time < config.EVENT_BURST_END)]
fit_window = burst[(burst.time >= config.MOTION_FIT_START) & (burst.time < config.MOTION_FIT_END)]

fig, ax = plt.subplots(figsize=(7, 5.5))
sc = ax.scatter(burst.cx, burst.cy, c=(burst.time - burst.time.iloc[0]).dt.total_seconds() / 60,
                 cmap=figures.SEQUENTIAL_CMAP, s=18, label="full 05:00-08:00 burst")
ax.plot(fit_window.cx, fit_window.cy, color="#e34948", linewidth=2.2, zorder=5,
        label="isolated single-cell passage (used for the motion fit)")
cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03); cbar.set_label("minutes into burst", fontsize=9)
ax.set_title("Rainfall centroid trajectory: multi-cellular, not one coherent storm", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout()
"""))

cells.append(code("""\
motion = adv.estimate_motion(cube)
print(f"speed={motion.speed_ms:.2f} m/s ({motion.speed_ms*3.6:.1f} km/hr)  bearing={motion.bearing_deg:.1f}° "
      f"(due east)  R²(along-track)={motion.r2_x:.2f}")

fp = adv.build_footprint(cube, motion)
val = adv.validate_footprint(cube, motion, fp)
print(f"footprint reconstruction R² across 58 pixels: median={val.r2.median():.2f} "
      f"(the storm's translation is well-determined; per-pixel detail is noisier since real cells "
      f"grow and decay, not just translate)")

fig, ax = plt.subplots(figsize=(6, 4.5))
im = ax.pcolormesh(fp.u_centers, fp.w_centers, fp.F, cmap=figures.SEQUENTIAL_CMAP, shading="auto")
ax.set_xlabel("along-track distance from storm center (m)"); ax.set_ylabel("cross-track distance (m)")
ax.set_title("Reconstructed storm footprint (storm-relative frame)", loc="left", fontsize=11)
cbar = fig.colorbar(im, ax=ax); cbar.set_label("mm/hr", fontsize=9)
fig.tight_layout()
"""))

cells.append(code("""\
pixels = radar.discover_pixels()
gage_xy = np.array([[p.x_center, p.y_center] for p in pixels])
gage_ids = [f"px{p.pixel_id}" for p in pixels]

catalogue, values = adv.build_realizations(cube, motion, fp, gage_xy, gage_ids)
display(catalogue[["realization_id", "direction_label", "domain_total_mm", "renorm_factor",
                    "peak_centroid_x", "peak_centroid_y", "event_centroid_x", "event_centroid_y"]]
        .round(2))

# every realization must deliver (post-renormalization) the same total rainfall as the baseline --
# this is the controlled variable that isolates "path" as the only thing that varies.
assert np.allclose(catalogue.domain_total_mm, catalogue.domain_total_mm.iloc[0], rtol=0.01), \\
    "renormalization failed to equalize domain-total rainfall across realizations"
print("OK: all 9 realizations deliver the same domain-total rainfall (±1%).")
"""))

# ---------------------------------------------------------------------------
cells.append(md("""\
## Part 4. Draping a surface over the pipes

Everything so far lives underground. Rain lands on a subcatchment, runs into a pipe, and if that pipe fills past
capacity the water has nowhere to go. Real flooding happens when water pushes up out of a manhole and spreads across
streets and gardens, so the model needs a ground surface for it to spread over.

That surface is a mesh, which is a sheet of triangles laid over the terrain with each triangle carrying its own
water depth. Picture chain mail draped across a landscape, fine enough to follow the shape of the ground and coarse
enough to compute. Elevation comes from the supplied SRTM data, resampled onto an exact 30 meter grid in the model's
own coordinate system, which gives roughly 23,000 triangles across the catchment.

Two pieces of software share this job and the division is worth understanding. The `mesh.py` module in this project
decides where triangles go, reading the elevation raster and emitting a list of corner points along with the
triangles that connect them. The engine's own `MeshBuilder` then turns that raw list into something a solver can
use. It works out which triangles share an edge, so water knows where it is allowed to flow. It identifies the outer
edges where water leaves the model. It computes each triangle's area, center and outward facing direction, and it
checks the whole assembly for degenerate geometry before anything runs. A pile of triangles only becomes a mesh once
that stitching has happened.

The final connection is the manholes. Each junction in the pipe network is tied to the triangle sitting above it,
which turns it into a two way door. Water surcharging from a full pipe arrives on the surface, and surface water
running over a manhole drains back down. For that exchange to mean anything, the ground elevation and the network's
own recorded manhole rim elevations have to agree about where the ground is, so we compare them and find a median
difference under a meter, which is close enough to trust.

One consequence of a 30 meter grid deserves attention when reading later results. A cell that size averages over
whole buildings and the streets between them, so this model cannot tell you that a particular road floods to the
curb. It can tell you that one district floods more than another under a given storm. Read the depths as a
comparison between trajectories rather than as survey data.
"""))

cells.append(code("""\
mesh = mesh_mod.build_mesh(nodes, poly_xy, force=FORCE_RERUN)
print(f"mesh: {len(mesh.triangles):,} triangles, {len(mesh.vertices):,} vertices, "
      f"{len(mesh.boundary_edges):,} boundary edges, {len(mesh.triangle_node_map)}/{len(nodes)} nodes coupled")
print(f"DEM vs. network rim elevations: median offset {mesh.dem_minus_rim_median_m:+.2f} m")

figures.mesh_figure(mesh, nodes)
plt.show()
"""))

cells.append(code("""\
base_sections = model.build_base_sections(inp_text, mesh, gage_xy, gage_ids)
print("base .inp assembled: 58 radar-pixel gages feed both 1D and 2D; ponding routed to the surface mesh")
"""))

# ---------------------------------------------------------------------------
cells.append(md("""\
## Part 5. One test run before committing to nine

The full comparison takes several minutes, so it is worth confirming the model behaves before spending them.

The most useful check is continuity, which is really just accounting. All the water that entered the model has to
equal the water that left, plus whatever is still sitting inside it. An engine that quietly loses or invents water
will still produce confident looking flood maps, and continuity error is how you catch that before you believe them.
Anything under a few percent is healthy, and the numbers here come in far below that.

We also confirm the pipes and the surface are genuinely talking to each other by checking that water moved in both
directions through the manhole connections, and we time the run so we know what nine of them will cost.
"""))

cells.append(code("""\
from openswmm.engine import Solver, HAS_2D, Statistics, MassBalance
import time

baseline_path = model.write_realization_inp(base_sections, "R00_baseline", gage_ids,
                                             values["R00_baseline"].index, values["R00_baseline"].values)

t0 = time.time()
with Solver(str(baseline_path)) as s:
    assert HAS_2D and s.surface2d.is_active, "2D module did not activate"
    surf = s.surface2d
    for _ in s.steps():
        pass
    wall_s = time.time() - t0
    mb1d, stats, mb2d = MassBalance(s), Statistics(s), surf.get_mass_balance()

    # MassBalance/Statistics properties are LIVE queries against the engine
    # handle, not cached snapshots -- capture plain floats before `with`
    # exits and destroys the handle (a BadHandleError follows any read
    # attempted after that point).
    runoff_err = mb1d.runoff_continuity_error
    routing_err = mb1d.routing_continuity_error
    continuity_2d = mb2d["continuity_error"]
    node_flooded_total = float(stats.node_vol_flooded.sum())
    n_flooded = int((stats.node_vol_flooded > 0).sum())

    print(f"wall clock: {wall_s:.1f} s for the full 6-hour simulation")
    print(f"1D runoff continuity error:  {runoff_err:.4f} %")
    print(f"1D routing continuity error: {routing_err:.4f} %")
    print(f"2D continuity error:         {continuity_2d:.2e}")
    print(f"2D rainfall in:              {mb2d['rainfall_in']:.0f} m^3")
    print(f"2D coupling 1D->2D (surcharge to surface): {mb2d['coupling_1d_to_2d_in']:.0f} m^3")
    print(f"2D coupling 2D->1D (surface draining back): {mb2d['coupling_2d_to_1d_out']:.0f} m^3")
    print(f"2D boundary outflow:         {mb2d['boundary_out']:.0f} m^3")
    print(f"1D node flooded volume (sum): {node_flooded_total:.0f} m^3 across {n_flooded} nodes")

assert abs(runoff_err) < 5 and abs(routing_err) < 5, "1D continuity gate failed"
assert abs(continuity_2d) < 0.05, "2D continuity gate failed"
print("\\nAll smoke-test gates passed.")
print(f"Projected ensemble time at {config.N_WORKERS} workers: "
      f"~{9 / config.N_WORKERS * wall_s:.0f} s")
"""))

# ---------------------------------------------------------------------------
cells.append(md("""\
## Part 6. Running all nine storms

Each of the nine trajectories now gets a full simulation. Results are cached, so re-running this notebook reuses
completed work unless you set `FORCE_RERUN` to True.

One practical note on how these are run. The obvious way to speed this up is to launch several simulations as
separate operating system processes at once. On this machine that crashed reliably, taking down a couple of workers
partway through and leaving the rest waiting forever on results that were never coming. The cause sits in a shared
native library during startup rather than in the model, and a single simulation on its own never has trouble.
Rather than fight it, this notebook runs the nine one after another. Part 7 gets its concurrency a different way, by
holding several engines inside a single process, which avoids the problem entirely.
"""))

cells.append(code("""\
t0 = time.time()
metrics = run.run_ensemble(base_sections, catalogue, values, n_workers=config.N_WORKERS, force=FORCE_RERUN)
print(f"ensemble wall time: {time.time() - t0:.1f} s")

cols = ["realization_id", "direction_label", "wall_s", "runoff_continuity_error_pct",
        "routing_continuity_error_pct", "continuity_error_2d", "flood_volume_2d_m3",
        "max_depth_2d_m", "node_vol_flooded_total_m3", "n_nodes_flooded"]
display(metrics[cols].round(3))

flagged = metrics[(metrics.runoff_continuity_error_pct.abs() > 5)
                   | (metrics.routing_continuity_error_pct.abs() > 5)
                   | (metrics.continuity_error_2d.abs() > 0.05)]
if len(flagged):
    print("\\nWARNING -- continuity outside gate thresholds for:", flagged.realization_id.tolist())
else:
    print("\\nAll 9 runs within continuity gate thresholds.")
"""))

# ---------------------------------------------------------------------------
cells.append(md("""\
## Part 7. Six storms, six engines, one process

Everything up to this point has been sequential. Open a simulation, run it to the end, close it, start the next one.
That is the ordinary way to work, and it means the only way to compare two storms is to finish both and inspect the
wreckage afterward.

Here we do something different. Six trajectories are opened at once, each with its own `Solver`, and all six are
held open together in this notebook's memory. Then we walk down the row of them exactly like the grandmaster at the
simultaneous exhibition, advancing each one by fifteen simulated minutes before moving on to the next, twenty four
times over. Every panel in the animation below is therefore showing the same moment of the same storm event playing
out six different ways.

The six were chosen to span the range of outcomes, from the trajectory that floods the catchment least to the one
that floods it most, with the observed storm included as a reference. Each panel shows water spreading across the
2D surface, and each title counts how many manholes have flooded so far, so you can watch the underground network
and the street surface fill together.

Watching them side by side reveals something the summary statistics cannot. These storms are not milder and worse
versions of one another. They put water in different places, and they put it there at different times, and timing
governs whether a pipe has drained enough to accept the next surge or is still full when it arrives.

A note on the shape of the experiment. All nine trajectories are built to pass through the same point at the same
moment, the observed storm's peak. They approach from six directions, converge on that instant, and separate again
afterward. Any apparent agreement in the middle of the animation is therefore built into the design and should not
be read as a result.

If you are reading this on GitHub rather than in a running Jupyter session, the interactive animation below will
appear blank, because GitHub strips the JavaScript that drives it. The same animation is saved as a GIF, which does
display:

![Six trajectories advancing together](media/trajectory_animation.gif)

This section also cost a bug worth recording. Opening a second `Solver` while the first was still open crashed the
process immediately, but only when the mapping library used elsewhere in this notebook had been loaded first.
Running one simulation at a time was never affected, which is why nothing earlier in the notebook noticed. The fix
is an import ordering rule in the setup cell, and `run.run_live_multi_engine` now checks for the bad order and
raises a clear error rather than dying silently.
"""))

cells.append(code("""\
# Six bearings spanning the outcome range, chosen from the already-computed
# 25-bearing sweep's results (cache/sweep_run.log): the calmest (270 deg),
# a low case (090), a mid-high case (060), a high case (000), and the most
# severe (165), alongside the observed baseline.
live_bearings = {b: l for b, l in adv.compass_bearings(24).items() if b in {0.0, 60.0, 90.0, 165.0, 270.0}}
catalogue_live, values_live = adv.build_realizations(
    cube, motion, fp, gage_xy, gage_ids, bearings=live_bearings, id_prefix="L")

live_ids = catalogue_live.realization_id.tolist()
for rid in live_ids:
    row = values_live[rid]
    model.write_realization_inp(base_sections, rid, gage_ids, row.index, row.values)
print(f"{len(live_ids)} trajectories staged for the live multi-engine run: {live_ids}")
"""))

cells.append(code("""\
live_labels = {row.realization_id: ("observed (baseline)" if pd.isna(row.bearing_deg)
                                    else f"{row.bearing_deg:03.0f}°")
              for _, row in catalogue_live.iterrows()}

t0 = time.time()
node_names_live, live_frames = run.run_live_multi_engine(live_ids)
print(f"{len(live_ids)} engines held open simultaneously, "
      f"{len(live_frames)} shared checkpoints, {time.time() - t0:.0f}s wall time")
"""))

cells.append(code("""\
from IPython.display import HTML

live_anim = figures.live_multi_engine_grid_animation(node_names_live, live_frames, mesh, live_labels)
HTML(live_anim.to_jshtml())
"""))

# ---------------------------------------------------------------------------
cells.append(md("""\
## Part 8. Comparing the nine outcomes

The animation showed character. These plots show magnitude.

Each box summarizes how flooding is distributed across space within a single trajectory, gathering every manhole
that flooded or every wet triangle on the surface, with a diamond marking the catchment wide total. Reading across
the boxes tells you whether a given storm direction produces a handful of severe local failures or widespread
shallow nuisance flooding, which are very different problems for whoever has to manage them and call for different
interventions.

The spread between the mildest and worst trajectory is the number to carry away. Every one of these is the same
storm delivering the same water. If a design standard specifies only a depth and a duration with no direction
attached, the resulting estimate of flood volume can land anywhere inside that range depending on a factor the
design never considered.
"""))

cells.append(code("""\
fig1 = figures.flood_volume_boxplot(metrics, catalogue)
fig2 = figures.max_depth_boxplot(metrics, catalogue)
fig3 = figures.peak_flooding_rate_boxplot(metrics, catalogue)
plt.show()
"""))

# ---------------------------------------------------------------------------
cells.append(md("""\
## What to take from this

Every storm in this notebook carried the same water. The rainfall totals were normalized to within one percent of
each other precisely so that nothing else could account for the results. Whatever difference appears in the flood
volumes came from the direction the storm traveled.

That carries a direct implication for practice. A design storm has no direction, so it cannot express this effect at
all, and a drainage system signed off against one may be considerably more exposed to some storm tracks than the
design run implies. Where that exposure sits depends on how the storm's path interacts with the network's own
drainage direction, which is the kind of thing that has to be simulated rather than reasoned about from a map.

A few limits are worth carrying forward. The terrain is represented at 30 meters with no buildings, so these depths
compare trajectories against each other and should not be read as street level predictions. The frozen field
reconstruction captures the storm's movement well and its internal evolution poorly, quantified by the R squared
values in Part 3. And this is one storm over one catchment, so the size of the effect found here does not transfer
to another site, although the existence of the effect is worth checking wherever you work.
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "openswmm2d", "language": "python", "name": "openswmm2d"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(nb, indent=1))
print("wrote", NB_PATH, NB_PATH.stat().st_size, "bytes,", len(cells), "cells")
