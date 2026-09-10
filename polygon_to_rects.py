"""
Convert a rectilinear yard polygon into the (box, obstacles) form that
max_x_path / build_population expect.

The polygon's own edges induce a grid. Any grid cell whose centre lies
outside the polygon is unreachable, so it becomes a forbidden rectangle.
Exact for rectilinear outlines; approximate for anything else.
"""

import numpy as np
from matplotlib.path import Path as MPath


def polygon_to_box_obs(P, merge=True):
    """
    P : (n, 2) polygon vertices, in order.

    Returns (box, obs) with
        box = [xmin, ymin, xmax, ymax]
        obs = (M, 4) rows [xmin, ymin, xmax, ymax]   the region inside the
              bounding box but outside the polygon
    """
    P = np.asarray(P, float)
    xs = np.unique(P[:, 0])
    ys = np.unique(P[:, 1])
    box = [xs.min(), ys.min(), xs.max(), ys.max()]

    poly = MPath(P)
    cx = (xs[:-1] + xs[1:]) / 2
    cy = (ys[:-1] + ys[1:]) / 2
    CX, CY = np.meshgrid(cx, cy, indexing="ij")
    inside = poly.contains_points(np.column_stack([CX.ravel(), CY.ravel()]))
    blocked = ~inside.reshape(len(cx), len(cy))

    if not merge:
        obs = [[xs[i], ys[j], xs[i + 1], ys[j + 1]]
               for i in range(len(cx)) for j in range(len(cy)) if blocked[i, j]]
        return box, np.array(obs, float).reshape(-1, 4)

    # greedy merge of blocked cells into maximal rectangles, column strips first
    used = np.zeros_like(blocked)
    obs = []
    for i in range(len(cx)):
        j = 0
        while j < len(cy):
            if blocked[i, j] and not used[i, j]:
                j2 = j
                while j2 + 1 < len(cy) and blocked[i, j2 + 1] and not used[i, j2 + 1]:
                    j2 += 1
                i2 = i
                while (i2 + 1 < len(cx)
                       and blocked[i2 + 1, j:j2 + 1].all()
                       and not used[i2 + 1, j:j2 + 1].any()):
                    i2 += 1
                used[i:i2 + 1, j:j2 + 1] = True
                obs.append([xs[i], ys[j], xs[i2 + 1], ys[j2 + 1]])
                j = j2 + 1
            else:
                j += 1
    return box, np.array(obs, float).reshape(-1, 4)


if __name__ == "__main__":
    raw = [(1488100,793400),(1488100,817700),(1580829,817700),(1580829,813229),
           (1628171,813229),(1628171,817700),(1672282,817700),(1672282,811556),
           (1694804,811556),(1694804,817700),(1906500,817700),(1906500,812501),
           (1776371,812501),(1776371,813171),(1727829,813171),(1727829,812501),
           (1711324,812501),(1711324,799371),(1662229,799371),(1662229,795028),
           (1660571,795028),(1660571,799371),(1615329,799371),(1615329,795028),
           (1615234,795028),(1615234,795093),(1596333,795093),(1596333,793400)]
    P = np.array(raw, float) / 1000.0

    box, obs = polygon_to_box_obs(P)
    np.set_printoptions(suppress=True)
    print("box:", np.round(box, 3))
    print(f"obstacles: {len(obs)} rectangles")
    print(np.round(obs, 3))