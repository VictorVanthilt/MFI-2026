"""
Overlap regions between the rectangles of a path, and random waypoints
sampled inside them (seed population for a GA over travel time).

Everything is driven by the numerical inputs box, obs, p_start, p_goal, Lmin.
Requires max_x_path.py in the same folder.

Rectangle conventions, kept explicit because the MATLAB mixed two:
    box, obs   : [x_min, y_min, x_max, y_max]
    R, ovl     : [x_min, x_max, y_min, y_max]   <- note the ordering
"""

import numpy as np

from max_x_path import max_x_path

rng = np.random.default_rng()


def overlaps(R, consecutive=True, tol=1e-9):
    """
    Intersections of the rectangles in R (n, 4), given as [x1 x2 y1 y2].

    consecutive=True  : only pairs (0,1), (1,2), (2,3), ... This is what a
                        path chain needs — the waypoints then come out in
                        travel order.
    consecutive=False : all pairs (a, b) with a < b, as in the MATLAB.

    Returns
        ovl   (m, 4) overlap rectangles [x1 x2 y1 y2]
        pairs (m, 2) 0-based indices of the two rectangles
        wh    (m, 2) width and height of each overlap
    Only intersections of strictly positive area are kept.
    """
    R = np.asarray(R, dtype=float)
    n = len(R)
    idx_pairs = ([(a, a + 1) for a in range(n - 1)] if consecutive
                 else [(a, b) for a in range(n - 1) for b in range(a + 1, n)])

    ovl, pairs, wh = [], [], []
    for a, b in idx_pairs:
        x1 = max(R[a, 0], R[b, 0])
        x2 = min(R[a, 1], R[b, 1])
        y1 = max(R[a, 2], R[b, 2])
        y2 = min(R[a, 3], R[b, 3])
        w, h = x2 - x1, y2 - y1
        if w <= tol or h <= tol:
            continue                          # touching along an edge, or disjoint
        ovl.append([x1, x2, y1, y2])
        pairs.append([a, b])
        wh.append([w, h])

    return (np.array(ovl).reshape(-1, 4),
            np.array(pairs, dtype=int).reshape(-1, 2),
            np.array(wh).reshape(-1, 2))


def sample_paths(ovl, n_paths=3):
    """
    One random waypoint inside each overlap region, for each of n_paths.

    Returns (n_paths, 2*m): row i is [x0 y0 x1 y1 ...], the waypoints of
    candidate path i in the order the overlaps are given.
    """
    ovl = np.asarray(ovl, dtype=float)
    m = len(ovl)
    if m == 0:
        return np.zeros((n_paths, 0))

    x = ovl[:, 0] + (ovl[:, 1] - ovl[:, 0]) * rng.random((n_paths, m))
    y = ovl[:, 2] + (ovl[:, 3] - ovl[:, 2]) * rng.random((n_paths, m))

    path = np.empty((n_paths, 2 * m))
    path[:, 0::2] = x
    path[:, 1::2] = y
    return path


def as_waypoints(path_row, p_start, p_goal):
    """Turn one row of sample_paths into a (k, 2) polyline from start to goal."""
    mid = np.asarray(path_row, float).reshape(-1, 2)
    return np.vstack([np.asarray(p_start, float),
                      mid,
                      np.asarray(p_goal, float)])


def gene_bounds(ovl):
    """
    Per-gene lower/upper bounds matching the [x0 y0 x1 y1 ...] layout,
    for a GA that must keep every waypoint inside its own overlap region.

    Returns (lo, hi), each of length 2*m.
    """
    ovl = np.asarray(ovl, dtype=float)
    m = len(ovl)
    lo, hi = np.empty(2 * m), np.empty(2 * m)
    lo[0::2], hi[0::2] = ovl[:, 0], ovl[:, 1]
    lo[1::2], hi[1::2] = ovl[:, 2], ovl[:, 3]
    return lo, hi


def build_population(box, obs, p_start, p_goal, Lmin=0.0,
                     n_paths=3, consecutive=True):
    """
    Full pipeline from the raw geometry.

    Parameters
    ----------
    box     : [x_min, y_min, x_max, y_max]                the yard
    obs     : (M, 4) rows [x_min, y_min, x_max, y_max]    forbidden zones
    p_start,
    p_goal  : [x, y]
    Lmin    : minimum admissible x-edge length
    n_paths : population size
    consecutive : restrict overlaps to consecutive rectangles of the chain

    Returns a dict with the path rectangles, the overlap regions, the
    sampled population, and the per-gene bounds for the GA.
    """
    path_rects, cover, all_rects = max_x_path(box, obs, p_start, p_goal, Lmin)
    ovl, pairs, wh = overlaps(path_rects, consecutive=consecutive)

    if consecutive and len(ovl) < len(path_rects) - 1:
        raise ValueError(
            "Some consecutive rectangles in the chain share only an edge "
            "(zero-area contact), so no waypoint can be sampled there. "
            "The BFS in max_x_path allows edge contact; this sampler does not.")

    P = sample_paths(ovl, n_paths)
    lo, hi = gene_bounds(ovl)

    return {"path_rects": path_rects,
            "cover": cover,
            "all_rects": all_rects,
            "ovl": ovl,
            "pairs": pairs,
            "wh": wh,
            "population": P,
            "lo": lo,
            "hi": hi}


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    # ---- numerical inputs, change these only -------------------------
    #box = [0, 0, 60, 16]                    # the yard
    #obs = [[26, 2, 32, 10],                 # forbidden zones
        #   [42, 8, 50, 12]]
    #p_start = [0, 12]
    #p_goal = [58, 6]
    #Lmin = 20.0
    #n_paths = 3
    # ------------------------------------------------------------------

    out = build_population(box, obs, p_start, p_goal, Lmin, n_paths)

    print("path_rects [x1 x2 y1 y2]:\n", out["path_rects"])
    print("\noverlap rects:\n", out["ovl"])
    print("pairs (0-based):", out["pairs"].tolist())
    print("width, height  :", out["wh"].tolist())

    P = out["population"]
    print("\npopulation (%d x %d):\n" % P.shape, np.round(P, 3))
    print("\ngene lower bounds:", np.round(out["lo"], 3))
    print("gene upper bounds:", np.round(out["hi"], 3))

    ok = np.all((P >= out["lo"] - 1e-9) & (P <= out["hi"] + 1e-9))
    print("\nall sampled points inside their overlap:", bool(ok))

    print("\nfirst candidate as a polyline:\n",
          np.round(as_waypoints(P[0], p_start, p_goal), 3))


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------
def print_overlap_table(pairs, ovl, wh):
    """Translation of the fprintf table. Indices are 0-based here."""
    print(f"{'a':>4s} {'b':>4s} {'x1':>8s} {'x2':>8s} "
          f"{'y1':>8s} {'y2':>8s} {'w':>8s} {'h':>8s}")
    for (a, b), r, (w, h) in zip(pairs, ovl, wh):
        print(f"{a:4d} {b:4d} {r[0]:8.4g} {r[1]:8.4g} "
              f"{r[2]:8.4g} {r[3]:8.4g} {w:8.4g} {h:8.4g}")


def plot_layout(box, obs, p_start, p_goal, path_rects, ovl, Lmin,
                waypoints=None, ax=None, save=None):
    """
    Yard, forbidden zones, path rectangles (dark blue) and their
    overlap regions (red).

    waypoints : optional (k, 2) polyline from as_waypoints(), drawn on top.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    box = np.asarray(box, float)
    obs = np.asarray(obs, float).reshape(-1, 4)
    path_rects = np.asarray(path_rects, float).reshape(-1, 4)
    ovl = np.asarray(ovl, float).reshape(-1, 4)

    RED = "#d1242f"        # forbidden zones
    BLUE = "#12355b"       # chosen maximal rectangles
    GREEN = "#1a9850"      # overlap regions

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.2))
    ax.set_aspect("equal")
    ax.set_xlim(box[0], box[2])
    ax.set_ylim(box[1], box[3])

    # field
    ax.add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                           facecolor="#f0f0f0", edgecolor="k", linewidth=1.2))

    # forbidden zones
    for k in range(len(obs)):
        ax.add_patch(Rectangle((obs[k, 0], obs[k, 1]),
                               obs[k, 2] - obs[k, 0], obs[k, 3] - obs[k, 1],
                               facecolor=RED, edgecolor="k", zorder=2,
                               label="forbidden" if k == 0 else None))

    # path rectangles, all dark blue  (r = [x1 x2 y1 y2])
    for k, r in enumerate(path_rects):
        ax.add_patch(Rectangle((r[0], r[2]), r[1] - r[0], r[3] - r[2],
                               facecolor=BLUE, alpha=0.25,
                               edgecolor=BLUE, linewidth=2, zorder=3,
                               label="path rect" if k == 0 else None))
        ax.text(r[[0, 1]].mean(), r[[2, 3]].mean(),
                f"{k}  ($L_x$={r[1] - r[0]:.4g})",
                ha="center", va="center", fontweight="bold",
                color=BLUE, zorder=6)

    # overlap regions, red, drawn on top
    for k, r in enumerate(ovl):
        ax.add_patch(Rectangle((r[0], r[2]), r[1] - r[0], r[3] - r[2],
                               facecolor=GREEN, alpha=0.60,
                               edgecolor=GREEN, linewidth=2, zorder=4,
                               label="overlap" if k == 0 else None))

    if waypoints is not None:
        wp = np.asarray(waypoints, float)
        ax.plot(wp[:, 0], wp[:, 1], "-o", color="k", linewidth=1.5,
                markersize=4, zorder=7, label="candidate path")

    ax.plot(*p_start, "o", color="w", markersize=10,
            markeredgecolor="k", markeredgewidth=1.5, zorder=8)
    ax.plot(*p_goal, "s", color="k", markersize=10,
            markeredgecolor="k", zorder=8)
    ax.text(p_start[0], p_start[1] - 1.4, r"$P_{start}$", ha="center", zorder=8)
    ax.text(p_goal[0], p_goal[1] - 1.4, r"$P_{goal}$", ha="center", zorder=8)

    ax.set_title(f"Path uses {len(path_rects)} rectangles, "
                 f"$L_{{min}}$ = {Lmin:g}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=4, frameon=False)

    if save:
        ax.figure.savefig(save, dpi=150, bbox_inches="tight")
    return ax