import sys
import time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from swmm_storm import inp_utils as iu, mesh as mesh_mod, model, radar, advection as adv, run, config


def main():
    text = open(config.INP_PATH, errors="ignore").read()
    sections = iu.split_sections(text)
    nodes = iu.read_node_table(sections)
    poly = iu.read_polygon_vertices(sections)
    m = mesh_mod.build_mesh(nodes, poly)

    cube = radar.build_event_cube()
    motion = adv.estimate_motion(cube)
    fp = adv.build_footprint(cube, motion)

    pixels = radar.discover_pixels()
    gage_xy = np.array([[p.x_center, p.y_center] for p in pixels])
    gage_ids = [f"px{p.pixel_id}" for p in pixels]

    base_sections = model.build_base_sections(text, m, gage_xy, gage_ids)
    catalogue, values = adv.build_realizations(cube, motion, fp, gage_xy, gage_ids)

    t0 = time.time()
    metrics = run.run_ensemble(base_sections, catalogue, values, n_workers=config.N_WORKERS, force=True)
    print("total ensemble wall time:", time.time() - t0, "s")

    pd.set_option("display.width", 200)
    cols = ["realization_id", "direction_label", "wall_s", "runoff_continuity_error_pct",
            "routing_continuity_error_pct", "continuity_error_2d", "flood_volume_2d_m3",
            "max_depth_2d_m", "node_vol_flooded_total_m3", "n_nodes_flooded"]
    print(metrics[cols].to_string())


if __name__ == "__main__":
    main()
