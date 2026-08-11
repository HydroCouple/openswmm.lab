"""Minimal, surgical .inp reader/writer.

We do not build a full object model of the Bellinge input file (17,675
lines). Instead we split it into named `[SECTION]` blocks, keep each
block's raw text, and let `mesh.py` / `model.py` splice specific blocks
in or out. This is robust to the file's existing formatting and quirks
and keeps every unrelated section byte-for-byte untouched.
"""
from __future__ import annotations

import re
from collections import OrderedDict

import numpy as np
import pandas as pd

_SECTION_RE = re.compile(r"^\[([A-Za-z0-9_]+)\]\s*$", re.MULTILINE)


def split_sections(text: str) -> "OrderedDict[str, str]":
    """Split raw .inp text into {SECTION_NAME: body_text} in file order.

    `body_text` excludes the `[NAME]` header line itself and runs up to
    (not including) the next `[SECTION]` header.
    """
    matches = list(_SECTION_RE.finditer(text))
    sections: "OrderedDict[str, str]" = OrderedDict()
    for i, m in enumerate(matches):
        name = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[name] = text[start:end]
    return sections


def join_sections(sections: "OrderedDict[str, str]") -> str:
    return "".join(f"[{name}]{body}" for name, body in sections.items())


def tokenize(body: str) -> list[list[str]]:
    """Non-comment, non-blank lines of a section body, whitespace-split."""
    rows = []
    for line in body.splitlines():
        s = line.strip()
        if not s or s.startswith(";"):
            continue
        rows.append(s.split())
    return rows


def read_node_table(sections: "OrderedDict[str, str]", include_outfalls: bool = False) -> pd.DataFrame:
    """JUNCTIONS + STORAGE (+ OUTFALLS if `include_outfalls`), with rim
    elevation (invert + max depth) and (x, y) joined from COORDINATES.

    `include_outfalls` defaults False: `mesh.build_mesh` couples *every* row
    of this table to a 2D triangle, and coupling outfalls to the mesh was
    never part of the validated model -- callers that only need node
    positions (e.g. for plotting the engine's full node-array output, which
    does include outfalls) should pass `include_outfalls=True` explicitly.

    Every SWMM node kind carries `Name Elev` in columns 0-1, but MaxDepth is
    only column 2 for JUNCTIONS/STORAGE -- for OUTFALLS, column 2 is the
    boundary `Type` (e.g. "FREE"), not a number, so outfalls get
    maxdepth=0 (rim = elev) rather than reusing that column.
    """
    rows = []
    for kind in ("JUNCTIONS", "STORAGE"):
        for tok in tokenize(sections.get(kind, "")):
            name, elev, maxdepth = tok[0], float(tok[1]), float(tok[2])
            rows.append({"name": name, "kind": kind, "elev": elev,
                         "maxdepth": maxdepth, "rim": elev + maxdepth})
    if include_outfalls:
        for tok in tokenize(sections.get("OUTFALLS", "")):
            name, elev = tok[0], float(tok[1])
            rows.append({"name": name, "kind": "OUTFALLS", "elev": elev,
                         "maxdepth": 0.0, "rim": elev})
    nodes = pd.DataFrame(rows)

    coords = {tok[0]: (float(tok[1]), float(tok[2]))
              for tok in tokenize(sections.get("COORDINATES", ""))}
    nodes["x"] = nodes["name"].map(lambda n: coords.get(n, (np.nan, np.nan))[0])
    nodes["y"] = nodes["name"].map(lambda n: coords.get(n, (np.nan, np.nan))[1])
    return nodes.dropna(subset=["x", "y"]).reset_index(drop=True)


def read_polygon_vertices(sections: "OrderedDict[str, str]") -> np.ndarray:
    """(N, 2) array of every [Polygons] vertex -- used only to shape the
    mesh-retention buffer, not for exact subcatchment geometry."""
    rows = [(float(tok[1]), float(tok[2])) for tok in tokenize(sections.get("POLYGONS", ""))]
    return np.array(rows, dtype=float)


def read_subcatchment_outlets(sections: "OrderedDict[str, str]") -> pd.DataFrame:
    """Subcatchment name -> outlet node name (used to assign each
    subcatchment to its nearest rain gage by outlet location)."""
    rows = [{"subcatchment": tok[0], "outlet": tok[2]}
            for tok in tokenize(sections.get("SUBCATCHMENTS", ""))]
    return pd.DataFrame(rows)
