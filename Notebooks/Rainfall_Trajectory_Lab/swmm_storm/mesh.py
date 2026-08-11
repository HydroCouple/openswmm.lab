"""DEM -> triangle vertices/connectivity, boundary conditions, node coupling.

Division of labour with the engine: `MeshBuilder` (src/engine/2d/mesh/) owns
the finite-volume mesh construction -- edge-neighbor adjacency, boundary-edge
identification, per-cell area/centroid, per-edge length/midpoint/outward
normal, plus validation (index bounds, degenerate triangles, non-positive
area or Manning's n) and incremental Z-dependent recompute. What it does not
do is generate triangles over terrain; there is no DEM/raster path into it.

This module fills exactly that gap: read the supplied SRTM DEM (reprojected
from EPSG:4326 to the model's CRS, EPSG:25832), emit `[2D_VERTICES]` /
`[2D_TRIANGLES]`, and hand off. Boundary edges are derived here too, but only
to *name* them in `[2D_BOUNDARY_CONDITIONS]` so each can be assigned a BC
type -- the engine independently derives its own adjacency from the triangles.

Mesh construction is a structured-grid-to-TIN conversion, not a general
Delaunay triangulation: the DEM read is aligned to `config.MESH_RES_M`, each
2x2 block of pixel centers becomes a quad, and each retained quad becomes
two triangles (diagonal alternated in a checkerboard to avoid a directional
bias in the flux stencil). This keeps triangle quality and count entirely
predictable, which matters for a live-demo runtime budget.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from affine import Affine
from scipy.spatial import cKDTree

from . import config, inp_utils as iu

_GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR")


# ---------------------------------------------------------------------------
# DEM access
# ---------------------------------------------------------------------------
def read_dem_window(bbox_25832: tuple[float, float, float, float],
                     res_m: float, pad_m: float = 500.0,
                     dem_path=None):
    """Bilinear-reprojected DEM read, windowed to `bbox_25832` + `pad_m`.

    The supplied SRTM tif is EPSG:4326; this reprojects it to the model's CRS
    on a target grid built explicitly here, snapped to whole multiples of
    `res_m`.

    The target grid is constructed rather than delegated on purpose:
    `WarpedVRT(..., resolution=...)` silently does nothing (there is no such
    parameter -- it lands in **kwargs and is dropped), so GDAL falls back to
    its own suggested resolution. For this source that came out at 20.7 m
    instead of the requested 30 m, i.e. ~2.1x the intended cell count and
    ~2x the runtime of every 2D simulation downstream. Setting
    `dst_transform`/`width`/`height` explicitly is the only way to actually
    pin the cell size.

    Source nodata is mapped to NaN so voids are never bilinearly smeared into
    neighboring real elevations; the caller drops any quad touching one.

    Returns (Z (n_rows, n_cols) float32 with NaN voids, transform,
    x_centers, y_centers).
    """
    minx, miny, maxx, maxy = bbox_25832
    minx, miny, maxx, maxy = minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m
    # snap outward to whole multiples of res_m so the grid is reproducible
    minx = np.floor(minx / res_m) * res_m
    miny = np.floor(miny / res_m) * res_m
    maxx = np.ceil(maxx / res_m) * res_m
    maxy = np.ceil(maxy / res_m) * res_m
    n_cols = int(round((maxx - minx) / res_m))
    n_rows = int(round((maxy - miny) / res_m))
    transform = Affine.translation(minx, maxy) * Affine.scale(res_m, -res_m)

    path = str(dem_path or config.SUPPLIED_DEM_PATH)
    Z = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    with rasterio.Env(**_GDAL_ENV):
        with rasterio.open(path) as src:
            reproject(
                source=rasterio.band(src, 1), destination=Z,
                src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
                dst_transform=transform, dst_crs=config.CRS_MODEL, dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
    x_centers = transform.c + (np.arange(n_cols) + 0.5) * transform.a
    y_centers = transform.f + (np.arange(n_rows) + 0.5) * transform.e
    return Z, transform, x_centers, y_centers


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------
@dataclass
class MeshResult:
    vertices: np.ndarray            # (nV, 3) x, y, z (post elevation-reconciliation)
    triangles: np.ndarray           # (nT, 3) int vertex indices, CCW
    mannings_n: np.ndarray          # (nT,)
    boundary_edges: list[tuple[int, int]]     # (tri_idx, local_edge_idx)
    triangle_node_map: list[tuple[int, str]]  # (tri_idx, node_name) coupled pairs
    n_grid_rows: int
    n_grid_cols: int
    dem_minus_rim_median_m: float = 0.0  # sanity check: DEM vs SWMM rim elevations


def build_mesh(nodes: pd.DataFrame, polygon_xy: np.ndarray,
               res_m: float = config.MESH_RES_M,
               buffer_m: float = config.MESH_BUFFER_M,
               force: bool = False) -> MeshResult:
    if config.MESH_CACHE.exists() and not force:
        d = np.load(config.MESH_CACHE, allow_pickle=True)
        return MeshResult(
            vertices=d["vertices"], triangles=d["triangles"], mannings_n=d["mannings_n"],
            boundary_edges=[tuple(e) for e in d["boundary_edges"]],
            triangle_node_map=[(int(t), str(n)) for t, n in d["triangle_node_map"]],
            n_grid_rows=int(d["n_grid_rows"]), n_grid_cols=int(d["n_grid_cols"]),
            dem_minus_rim_median_m=float(d["dem_minus_rim_median_m"]),
        )

    bbox = (nodes.x.min(), nodes.y.min(), nodes.x.max(), nodes.y.max())
    Z, transform, xc, yc = read_dem_window(bbox, res_m)
    n_rows, n_cols = Z.shape
    xx, yy = np.meshgrid(xc, yc)  # (n_rows, n_cols)

    # --- retention mask: within buffer_m of any node or subcatchment vertex,
    #     and backed by real elevation (a NaN void can't carry a vertex Z, and
    #     any quad touching one is dropped rather than interpolated across).
    network_xy = np.vstack([nodes[["x", "y"]].values, polygon_xy])
    tree = cKDTree(network_xy)
    d, _ = tree.query(np.c_[xx.ravel(), yy.ravel()])
    vertex_ok = (d <= buffer_m).reshape(n_rows, n_cols) & np.isfinite(Z)

    # --- 2 triangles per retained quad, checkerboard diagonal
    vidx = np.arange(n_rows * n_cols).reshape(n_rows, n_cols)
    tris = []
    for i in range(n_rows - 1):
        for j in range(n_cols - 1):
            corners = (vidx[i, j], vidx[i, j + 1], vidx[i + 1, j], vidx[i + 1, j + 1])
            if not (vertex_ok[i, j] and vertex_ok[i, j + 1]
                    and vertex_ok[i + 1, j] and vertex_ok[i + 1, j + 1]):
                continue
            bl, br, tl, tr = corners
            if (i + j) % 2 == 0:
                tris.append((bl, br, tr))
                tris.append((bl, tr, tl))
            else:
                tris.append((bl, br, tl))
                tris.append((br, tr, tl))
    triangles = np.array(tris, dtype=np.int64)

    used = np.unique(triangles)
    remap = -np.ones(n_rows * n_cols, dtype=np.int64)
    remap[used] = np.arange(len(used))
    triangles = remap[triangles]

    vx = xx.ravel()[used]
    vy = yy.ravel()[used]
    vz_dem = Z.ravel()[used].astype(np.float64)

    # --- elevation reconciliation at coupled nodes (smooth, no shared-vertex
    #     conflict). Bellinge's junctions are dense (~995 over ~7 km of
    #     network), so influence radii from different nodes routinely
    #     overlap; correction must be a weighted AVERAGE of every node's
    #     target dz at a vertex, not a sum, or overlapping nodes stack and
    #     overcorrect.
    vtree = cKDTree(np.c_[vx, vy])
    num = np.zeros_like(vz_dem)
    den = np.zeros_like(vz_dem)
    radius = 3.0 * res_m
    for _, n in nodes.iterrows():
        _, i0 = vtree.query([n.x, n.y])
        dz = n.rim - vz_dem[i0]
        idx = vtree.query_ball_point([n.x, n.y], radius)
        if not idx:
            continue
        idx = np.array(idx)
        dd = np.hypot(vx[idx] - n.x, vy[idx] - n.y)
        w = np.clip(1.0 - dd / radius, 0.0, 1.0)
        num[idx] += w * dz
        den[idx] += w
    correction = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    vz = vz_dem + correction
    vertices = np.column_stack([vx, vy, vz])

    # --- boundary edges: any (tri, local_edge) whose vertex pair is shared
    #     by exactly one triangle. Local edge e is opposite vertex e
    #     (matches the engine's own [2D_BOUNDARY_CONDITIONS] convention).
    edge_count: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for ti, (a, b, c) in enumerate(triangles):
        for e, (p, q) in enumerate([(b, c), (c, a), (a, b)]):
            key = (p, q) if p < q else (q, p)
            edge_count.setdefault(key, []).append((ti, e))
    boundary_edges = [owners[0] for owners in edge_count.values() if len(owners) == 1]

    # --- couple every node to the triangle containing it (nearest centroid
    #     fallback if the point-in-triangle test misses due to fp edge cases)
    tri_xy = vertices[triangles][:, :, :2]  # (nT, 3, 2)
    ctree = cKDTree(tri_xy.mean(axis=1))
    triangle_node_map = []
    for _, n in nodes.iterrows():
        cand = ctree.query([n.x, n.y], k=8)[1]
        cand = np.atleast_1d(cand)
        hit = None
        for ti in cand:
            if _point_in_triangle((n.x, n.y), tri_xy[ti]):
                hit = ti
                break
        if hit is None:
            hit = int(cand[0])  # fallback: nearest centroid
        triangle_node_map.append((int(hit), n["name"]))

    mannings_n = np.full(len(triangles), config.MANNINGS_N_2D)

    # Sanity check retained from the old two-DEM cross-check: the raw DEM,
    # sampled at each node before any reconciliation, should sit close to
    # that node's SWMM rim elevation. A large median offset means the DEM and
    # the network disagree about the ground, which would invalidate coupling.
    dem_at_nodes = np.array([vz_dem[vtree.query([n.x, n.y])[1]] for _, n in nodes.iterrows()])
    dem_minus_rim_median_m = float(np.median(dem_at_nodes - nodes.rim.values))

    result = MeshResult(
        vertices=vertices, triangles=triangles, mannings_n=mannings_n,
        boundary_edges=boundary_edges, triangle_node_map=triangle_node_map,
        n_grid_rows=n_rows, n_grid_cols=n_cols,
        dem_minus_rim_median_m=dem_minus_rim_median_m,
    )
    np.savez_compressed(
        config.MESH_CACHE, vertices=vertices, triangles=triangles, mannings_n=mannings_n,
        boundary_edges=np.array(boundary_edges), triangle_node_map=np.array(triangle_node_map, dtype=object),
        n_grid_rows=n_rows, n_grid_cols=n_cols,
        dem_minus_rim_median_m=dem_minus_rim_median_m,
    )
    return result


def _point_in_triangle(p, tri_xy: np.ndarray) -> bool:
    """Barycentric sign test; tri_xy is (3, 2)."""
    (x, y) = p
    (x1, y1), (x2, y2), (x3, y3) = tri_xy
    d1 = (x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)
    d2 = (x - x3) * (y2 - y3) - (x2 - x3) * (y - y3)
    d3 = (x - x1) * (y3 - y1) - (x3 - x1) * (y - y1)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


# ---------------------------------------------------------------------------
# .inp section text for the mesh
# ---------------------------------------------------------------------------
def mesh_to_inp_sections(mesh: MeshResult, boundary_slope: float = 0.005,
                          report_2d: bool = False, output_file: str | None = None) -> dict[str, str]:
    lines = {}

    opt = (
        "\n;;Parameter             Value      ;; Notes\n"
        "MAX_TIMESTEP            10.0       ;; Max marcher step (s)\n"
        "DRY_DEPTH               0.001      ;; Wet/dry threshold (m)\n"
        "LIMITER_EPSILON         1.0e-6\n"
        "COUPLING_CD              " + str(config.COUPLING_CD) + "\n"
        "INTEGRATOR              EXPLICIT\n"
        "THETA                   0.8\n"
        "CFL_NUMBER               0.7\n"
        "H_MOVE                  0.003\n"
        "LTS_TIERS               4\n"
        "FROUDE_MAX               1.5\n"
        "COUPLING_AREA           DEFAULT\n"
        "RAINFALL_MODE           NATURAL_NEIGHBOUR\n"
        f"REPORT_2D                 {'YES' if report_2d else 'NO'}\n"
    )
    if report_2d and output_file:
        opt += f"OUTPUT_FILE              {output_file}\n"
    lines["2D_OPTIONS"] = opt

    v_lines = ["\n;;  X        Y        Z"]
    for x, y, z in mesh.vertices:
        v_lines.append(f"    {x:.3f}   {y:.3f}   {z:.3f}")
    lines["2D_VERTICES"] = "\n".join(v_lines) + "\n"

    t_lines = ["\n;; V1  V2  V3   MANNINGS_N"]
    for (a, b, c), n in zip(mesh.triangles, mesh.mannings_n):
        t_lines.append(f"   {a}   {b}   {c}   {n:.4f}")
    lines["2D_TRIANGLES"] = "\n".join(t_lines) + "\n"

    m_lines = ["\n;; TRIANGLE   SWMM_NODE   CD     AREA (m2)"]
    for ti, node_name in mesh.triangle_node_map:
        m_lines.append(f"   {ti}   {node_name}   {config.COUPLING_CD}   {config.COUPLING_AREA_M2}")
    lines["2D_TRIANGLE_NODE_MAP"] = "\n".join(m_lines) + "\n"

    b_lines = ["\n;; TRI   EDGE   TYPE          PARAM_1    PARAM_2   GROUP"]
    for ti, e in mesh.boundary_edges:
        b_lines.append(f"   {ti}   {e}   NORMAL_FLOW   {boundary_slope}   *   *")
    lines["2D_BOUNDARY_CONDITIONS"] = "\n".join(b_lines) + "\n"

    return lines
