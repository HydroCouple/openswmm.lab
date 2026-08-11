"""Box plots (spatial flood-metric distributions per trajectory) and
centroid scatter plots (storm path vs. flood outcome), styled per the
`dataviz` skill: fixed categorical hue order for the 8 compass bearings
(never cycled/reassigned), a single-hue sequential ramp for magnitude,
thin recessive marks, and a legend whenever more than one series is shown.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from scipy.spatial import cKDTree

from . import config, run as run_mod

# Fixed 8-hue categorical order (blue, orange, aqua, yellow, magenta, green,
# violet, red) assigned by compass bearing, never re-cycled. Baseline is the
# observed control, not a compass direction -- a neutral gray, not a 9th hue.
_CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
_BASELINE_COLOR = "#6b6a66"
_GRID_COLOR = "#d8d7d2"
_TEXT_SECONDARY = "#52514e"

_SEQUENTIAL_STEPS = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("seq_blue", _SEQUENTIAL_STEPS)


def _direction_order(catalogue: pd.DataFrame) -> list[str]:
    """Baseline first, then compass bearings in the fixed 0/45/.../315 order."""
    order = ["R00_baseline"]
    bearings = sorted(catalogue.loc[catalogue.bearing_deg.notna(), "bearing_deg"].unique())
    by_bearing = catalogue.set_index("bearing_deg")["realization_id"]
    order += [by_bearing[b] for b in bearings]
    return [r for r in order if r in set(catalogue.realization_id)]


def _color_for(realization_id: str, order: list[str]) -> str:
    if realization_id == "R00_baseline":
        return _BASELINE_COLOR
    idx = [r for r in order if r != "R00_baseline"].index(realization_id)
    return _CATEGORICAL[idx % len(_CATEGORICAL)]


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_GRID_COLOR)
    ax.spines["bottom"].set_color(_GRID_COLOR)
    ax.tick_params(colors=_TEXT_SECONDARY)
    ax.yaxis.grid(True, color=_GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _spatial_boxplot(
    metrics: pd.DataFrame, catalogue: pd.DataFrame,
    array_key: str, filter_fn, total_key: str | None,
    title: str, ylabel: str, log_scale: bool = False,
) -> plt.Figure:
    order = [r for r in _direction_order(catalogue) if r in set(metrics.realization_id)]
    label_map = catalogue.set_index("realization_id")["direction_label"]

    data, colors, labels, totals = [], [], [], []
    for rid in order:
        arrays = run_mod.load_arrays(rid)
        vals = arrays[array_key]
        vals = vals[filter_fn(vals)]
        data.append(vals)
        colors.append(_color_for(rid, order))
        labels.append(label_map.get(rid, rid))
        if total_key:
            totals.append(metrics.loc[metrics.realization_id == rid, total_key].iloc[0])

    # order left-to-right by median (baseline stays anchored first for reference)
    med = [np.median(d) if len(d) else 0.0 for d in data]
    perm = [0] + sorted(range(1, len(data)), key=lambda i: med[i])
    data, colors, labels = [data[i] for i in perm], [colors[i] for i in perm], [labels[i] for i in perm]
    if total_key:
        totals = [totals[i] for i in perm]

    fig, ax = plt.subplots(figsize=(9, 5))
    _style_axes(ax)
    bp = ax.boxplot(
        data, patch_artist=True, widths=0.55, showfliers=False,
        medianprops=dict(color="white", linewidth=1.6),
        whiskerprops=dict(color=_TEXT_SECONDARY, linewidth=1.0),
        capprops=dict(color=_TEXT_SECONDARY, linewidth=1.0),
        boxprops=dict(linewidth=0.8, edgecolor="white"),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.88)

    if total_key:
        ax.scatter(range(1, len(totals) + 1), totals, marker="D", s=46,
                   facecolor="none", edgecolor="#0b0b0b", linewidth=1.3,
                   zorder=5, label="system total")
        ax.legend(frameon=False, loc="upper left", fontsize=9)

    if log_scale:
        ax.set_yscale("log")
        ax.yaxis.grid(True, which="major", color=_GRID_COLOR, linewidth=0.8, zorder=0)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=0, fontsize=9, color=_TEXT_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=10, color=_TEXT_SECONDARY)
    ax.set_title(title, fontsize=12, color="#0b0b0b", loc="left", pad=12)
    fig.tight_layout()
    return fig


def flood_volume_boxplot(metrics: pd.DataFrame, catalogue: pd.DataFrame) -> plt.Figure:
    """Per-node cumulative flooded volume (flooding nodes only), one box per
    trajectory; diamond marker = system-wide total (`node_vol_flooded_total_m3`).
    Log-scaled: individual node volumes and the ~150-300-node system total
    span several orders of magnitude on the same axis, and a linear scale
    squashes the boxes to invisible slivers under the total markers."""
    return _spatial_boxplot(
        metrics, catalogue, array_key="node_vol_flooded",
        filter_fn=lambda v: v > 0, total_key="node_vol_flooded_total_m3",
        title="1D node flooding volume by storm trajectory",
        ylabel="Node flooded volume (m$^3$, log scale), flooding nodes only",
        log_scale=True,
    )


def max_depth_boxplot(metrics: pd.DataFrame, catalogue: pd.DataFrame) -> plt.Figure:
    """Per-triangle envelope max depth (wet cells only), one box per trajectory."""
    thresh = config.FLOOD_DEPTH_THRESHOLD_M
    return _spatial_boxplot(
        metrics, catalogue, array_key="tri_max_depth",
        filter_fn=lambda v: v > thresh, total_key="max_depth_2d_m",
        title="2D surface peak flood depth by storm trajectory",
        ylabel=f"Triangle max depth (m), cells > {thresh} m",
    )


def peak_flooding_rate_boxplot(metrics: pd.DataFrame, catalogue: pd.DataFrame) -> plt.Figure:
    """Per-node peak overflow (flooding) rate, flooding nodes only, one box
    per trajectory."""
    return _spatial_boxplot(
        metrics, catalogue, array_key="node_max_overflow",
        filter_fn=lambda v: v > 0, total_key=None,
        title="Peak node flooding rate by storm trajectory",
        ylabel="Node peak overflow rate (project flow units)",
    )


# ---------------------------------------------------------------------------
# True live multi-engine animation: N `Solver` handles held open at once in
# one process (run.run_live_multi_engine), stepped together to shared
# checkpoints. Every panel below is at the SAME simulated moment because
# they were advanced together -- the payoff of the engine's reentrant
# design, not N independently-completed runs replayed side by side.
# ---------------------------------------------------------------------------
def live_multi_engine_grid_animation(
    node_names: list[str], frames: list[dict], mesh, realization_labels: dict[str, str],
    n_cols: int = 3, interval_ms: int = 450,
) -> FuncAnimation:
    """Small-multiples grid, one filled 2D flood-depth surface per
    trajectory, all advancing through identical simulated-time checkpoints.
    Each panel's title also carries a live "N nodes flooded so far" count,
    so 1D (node) and 2D (overland) flooding are both visible together.
    """
    realization_ids = list(frames[0]["data"].keys())
    n = len(realization_ids)
    n_rows = -(-n // n_cols)  # ceil division

    triang = mtri.Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.triangles)
    global_max = max(frame["data"][rid]["tri_depths"].max()
                     for frame in frames for rid in realization_ids)
    # Log-scaled: most of the wet area is a few cm of shallow sheet flow,
    # with only rare, localized surcharge points reaching a meter or more.
    # A linear scale over that range makes the typical flooding invisible --
    # everything paints the same pale color except a handful of dark
    # specks. Depths are floored to 1 cm before display so dry cells (0 m,
    # otherwise undefined on a log scale) land at the palest step instead.
    depth_floor = 0.01
    norm = LogNorm(vmin=depth_floor, vmax=max(global_max, depth_floor * 10))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 4.2 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    tripcolors, titles = [], []
    zeros = np.full(len(mesh.triangles), depth_floor)
    for ax, rid in zip(axes, realization_ids):
        tp = ax.tripcolor(triang, facecolors=zeros, cmap=SEQUENTIAL_CMAP, norm=norm, edgecolors="none")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        ttl = ax.set_title(realization_labels.get(rid, rid), fontsize=9.5, color="#0b0b0b")
        tripcolors.append(tp); titles.append(ttl)
    for ax in axes[n:]:
        ax.axis("off")

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQUENTIAL_CMAP),
                        ax=axes.tolist(), fraction=0.025, pad=0.02, shrink=0.7)
    cbar.set_label("2D overland flood depth (m)", fontsize=9)
    cbar.outline.set_visible(False)

    suptitle = fig.suptitle("", fontsize=13, color="#0b0b0b", x=0.02, ha="left", y=0.99)

    def update(frame_i):
        frame = frames[frame_i]
        h = frame["elapsed"].total_seconds() / 3600.0
        suptitle.set_text(f"All {n} trajectories at the same simulated moment, t+{h:.2f} h")
        for tp, ttl, rid in zip(tripcolors, titles, realization_ids):
            d = frame["data"][rid]
            tp.set_array(np.maximum(d["tri_depths"], depth_floor))
            n_flooded = int((d["node_vol_flooded"] > 0).sum())
            label = realization_labels.get(rid, rid)
            ttl.set_text(f"{label}\n{n_flooded} nodes flooded so far")
        return tripcolors + titles + [suptitle]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=interval_ms,
                        blit=False, repeat=True)
    plt.close(fig)
    return anim


# ---------------------------------------------------------------------------
# Mesh figure -- what the terrain actually looks like once triangulated, and
# where the 1D network is stitched into it.
# ---------------------------------------------------------------------------
def mesh_figure(mesh, nodes: pd.DataFrame, zoom_half_width_m: float = 320.0) -> plt.Figure:
    """Two panels: the full triangulated surface colored by elevation, and a
    zoom on the densest patch of network showing individual triangle edges,
    the coupled junctions, and the retained-domain boundary.

    The zoom exists because at full extent 24k triangles at 30 m read as a
    smooth raster -- you cannot see that it *is* a mesh. The inset is where
    the triangulation, the checkerboard diagonals, and the node-to-triangle
    coupling become visible.
    """
    triang = mtri.Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.triangles)
    tri_z = mesh.vertices[mesh.triangles, 2].mean(axis=1)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6.2))

    # --- panel 0: full mesh, elevation-shaded
    tp = ax0.tripcolor(triang, facecolors=tri_z, cmap="terrain", edgecolors="none")
    ax0.scatter(nodes.x, nodes.y, s=1.5, color="#0b0b0b", alpha=0.5, zorder=3)
    cbar = fig.colorbar(tp, ax=ax0, fraction=0.04, pad=0.02)
    cbar.set_label("ground elevation (m)", fontsize=9)
    cbar.outline.set_visible(False)
    ax0.set_title(f"2D surface mesh: {len(mesh.triangles):,} triangles, "
                  f"{len(mesh.vertices):,} vertices",
                  fontsize=11, color="#0b0b0b", loc="left")

    # --- zoom window: centered on the densest cluster of coupled nodes
    coupled_names = {n for _, n in mesh.triangle_node_map}
    cn = nodes[nodes.name.isin(coupled_names)]
    ntree = cKDTree(cn[["x", "y"]].values)
    counts = np.array([len(ntree.query_ball_point(p, zoom_half_width_m))
                       for p in cn[["x", "y"]].values])
    cx, cy = cn[["x", "y"]].values[counts.argmax()]
    hw = zoom_half_width_m

    ax1.triplot(triang, color="#9a9992", linewidth=0.35, zorder=1)
    ax1.tripcolor(triang, facecolors=tri_z, cmap="terrain", edgecolors="none",
                  alpha=0.55, zorder=0)
    inz = cn[(cn.x.between(cx - hw, cx + hw)) & (cn.y.between(cy - hw, cy + hw))]
    ax1.scatter(inz.x, inz.y, s=26, color="#e34948", edgecolor="white",
                linewidth=0.7, zorder=4, label="coupled junction")
    ax1.set_xlim(cx - hw, cx + hw); ax1.set_ylim(cy - hw, cy + hw)
    ax1.legend(frameon=False, fontsize=9, loc="upper right")
    ax1.set_title(f"zoom: {2*hw:.0f} m across, showing individual triangles and "
                  f"1D/2D coupling points", fontsize=11, color="#0b0b0b", loc="left")

    for ax in (ax0, ax1):
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    return fig
