"""Assemble a per-realization Bellinge .inp: 58 radar-pixel rain gages
feeding both the 1D subcatchments and (via RAINFALL_MODE NATURAL_NEIGHBOUR)
the 2D mesh, plus the mesh sections from `mesh.py`.

The base file (everything except each realization's rainfall numbers) is
built once and reused; only the [TIMESERIES] block differs per realization,
so synthesizing all 9 .inp files is a cheap string-splice operation, not a
9x repeat of mesh generation or subcatchment reassignment.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import config, inp_utils as iu, mesh as mesh_mod


def assign_subcatchments_to_gages(sections: "OrderedDict[str, str]",
                                   gage_xy: np.ndarray, gage_ids: list[str]) -> dict[str, str]:
    """Nearest-gage assignment by each subcatchment's OUTLET node location
    (a subcatchment has no single point geometry of its own; its outlet is
    always present and is a reasonable positional proxy)."""
    coords = {tok[0]: (float(tok[1]), float(tok[2]))
              for tok in iu.tokenize(sections.get("COORDINATES", ""))}
    outlets = iu.read_subcatchment_outlets(sections)
    tree = cKDTree(gage_xy)

    mapping = {}
    missing = 0
    for _, row in outlets.iterrows():
        xy = coords.get(row.outlet)
        if xy is None:
            missing += 1
            mapping[row.subcatchment] = gage_ids[0]  # fallback: nearest overall
            continue
        _, i = tree.query(xy)
        mapping[row.subcatchment] = gage_ids[i]
    if missing:
        print(f"warning: {missing} subcatchment outlet(s) not found in COORDINATES; "
              f"assigned to {gage_ids[0]}")
    return mapping


def _set_option(body: str, key: str, value: str) -> str:
    """Replace a single `[OPTIONS]`-style `KEY   value` line, keeping the
    key's original column position; raises if the key isn't present."""
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if line.strip().upper().startswith(key.upper() + " ") or line.strip().upper() == key.upper():
            prefix_len = len(line) - len(line.lstrip())
            lines[i] = " " * prefix_len + f"{key:<21} {value}"
            return "\n".join(lines) + ("\n" if body.endswith("\n") else "")
    raise KeyError(f"OPTIONS key {key!r} not found")


def build_base_sections(
    inp_text: str, mesh: mesh_mod.MeshResult,
    gage_xy: np.ndarray, gage_ids: list[str],
) -> "OrderedDict[str, str]":
    sections = iu.split_sections(inp_text)

    # --- [OPTIONS]: ponding must exit through the mesh, not pool virtually
    #     at the node; trim the run to the event window.
    opt = sections["OPTIONS"]
    opt = _set_option(opt, "ALLOW_PONDING", "NO")
    opt = _set_option(opt, "START_DATE", config.SIM_START_DATE)
    opt = _set_option(opt, "START_TIME", config.SIM_START_TIME)
    opt = _set_option(opt, "REPORT_START_DATE", config.SIM_START_DATE)
    opt = _set_option(opt, "REPORT_START_TIME", config.SIM_START_TIME)
    opt = _set_option(opt, "END_DATE", config.SIM_END_DATE)
    opt = _set_option(opt, "END_TIME", config.SIM_END_TIME)
    opt = _set_option(opt, "REPORT_STEP", config.REPORT_STEP)
    sections["OPTIONS"] = opt

    # --- [RAINGAGES] + [SYMBOLS]: one INTENSITY/TIMESERIES gage per radar pixel
    rg_lines = ["\n;;Name   Format      Interval  SCF   Source"]
    sym_lines = ["\n;;Gage           X-Coord            Y-Coord"]
    for gid, (x, y) in zip(gage_ids, gage_xy):
        rg_lines.append(f"{gid:<10} INTENSITY   0:01      1.0   TIMESERIES {gid}_ts")
        sym_lines.append(f"{gid:<16} {x:.4f}        {y:.4f}")
    sections["RAINGAGES"] = "\n".join(rg_lines) + "\n"
    sections["SYMBOLS"] = "\n".join(sym_lines) + "\n"

    # --- [SUBCATCHMENTS]: reassign the RainGage column (token 1) to the
    #     nearest new gage; every other field is passed through unchanged.
    mapping = assign_subcatchments_to_gages(sections, gage_xy, gage_ids)
    sc_lines = ["\n;;Name  RainGage  Outlet  Area  %Imperv  Width  %Slope  CurbLen"]
    for tok in iu.tokenize(sections["SUBCATCHMENTS"]):
        name = tok[0]
        tok[1] = mapping.get(name, gage_ids[0])
        sc_lines.append(" ".join(tok))
    sections["SUBCATCHMENTS"] = "\n".join(sc_lines) + "\n"

    # --- [TIMESERIES]: placeholder, filled per realization
    sections["TIMESERIES"] = "\n;; filled per realization by render_timeseries_block()\n"

    # --- mesh sections (identical across all realizations)
    for name, body in mesh_mod.mesh_to_inp_sections(mesh).items():
        sections[name] = body

    return sections


def render_timeseries_block(gage_ids: list[str], times: pd.DatetimeIndex,
                             values: np.ndarray, start_date: str) -> str:
    """Run-length-compressed [TIMESERIES] body for all gages: an INTENSITY
    gage's rate holds from one listed time until the next, so only rows
    where a gage's value actually changes need to be written. This cuts a
    naive 360-row-per-gage x 58-gage block by >90% for a storm that's dry
    most of the 6-hour window.
    """
    lines = ["\n;;Name        Date        Time   Value"]
    date_str = pd.Timestamp(times[0]).strftime("%m/%d/%Y")
    for j, gid in enumerate(gage_ids):
        v = values[:, j]
        name = f"{gid}_ts"
        first = True
        prev = None
        for t, val in zip(times, v):
            val = round(float(val), 4)
            if prev is not None and val == prev:
                continue
            hhmm = pd.Timestamp(t).strftime("%H:%M")
            if first:
                lines.append(f"{name:<12} {date_str}   {hhmm}   {val}")
                first = False
            else:
                lines.append(f"{name:<12} {' ' * 10}   {hhmm}   {val}")
            prev = val
        if prev != 0.0:
            end_hhmm = pd.Timestamp(times[-1] + (times[-1] - times[-2])).strftime("%H:%M")
            lines.append(f"{name:<12} {' ' * 10}   {end_hhmm}   0.0")
    return "\n".join(lines) + "\n"


def write_realization_inp(
    base_sections: "OrderedDict[str, str]", realization_id: str,
    gage_ids: list[str], times: pd.DatetimeIndex, values: np.ndarray,
    out_dir: Path = config.RUNS_DIR,
) -> Path:
    sections = OrderedDict(base_sections)  # shallow copy; only TIMESERIES changes
    sections["TIMESERIES"] = render_timeseries_block(gage_ids, times, values,
                                                       config.SIM_START_DATE)
    run_dir = out_dir / realization_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{realization_id}.inp"
    path.write_text(iu.join_sections(sections))
    return path
