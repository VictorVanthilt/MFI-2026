"""
maxXPath: maximal free rectangles in a box with rectangular obstacles,
a greedy cover of the free region, and a chain of rectangles from start to goal.

Python translation of the MATLAB function of the same name.

Rectangle convention differs between input and output, as in the original:
    box, obs  : [xmin, ymin, xmax, ymax]
    pathRects, cover, allRects : [xmin, xmax, ymin, ymax]
"""

import warnings
import numpy as np


def max_x_path(box, obs, p_start, p_goal, Lmin=0.0, tol=1e-9):
    """
    Parameters
    ----------
    box     : [xmin, ymin, xmax, ymax]
    obs     : (M, 4) array, rows [xmin, ymin, xmax, ymax]
    p_start,
    p_goal  : [x, y]
    Lmin    : minimum admissible x-edge length (0 disables the filter)

    Returns
    -------
    path_rects : chain of rectangles from p_start to p_goal, [x1 x2 y1 y2]
    cover      : greedy cover of the free region, widest x-edge first
    all_rects  : all maximal free rectangles with x-edge >= Lmin
    """
    box = np.asarray(box, dtype=float)
    obs = np.asarray(obs, dtype=float).reshape(-1, 4)
    p_start = np.asarray(p_start, dtype=float)
    p_goal = np.asarray(p_goal, dtype=float)

    # ---- grid induced by all edges ------------------------------------
    xs = np.unique(np.concatenate([box[[0, 2]], obs[:, 0], obs[:, 2]]))
    ys = np.unique(np.concatenate([box[[1, 3]], obs[:, 1], obs[:, 3]]))
    xs = xs[(xs >= box[0]) & (xs <= box[2])]
    ys = ys[(ys >= box[1]) & (ys <= box[3])]

    nx, ny = len(xs) - 1, len(ys) - 1
    cx = (xs[:-1] + xs[1:]) / 2.0
    cy = (ys[:-1] + ys[1:]) / 2.0

    blocked = np.zeros((nx, ny), dtype=bool)
    for k in range(len(obs)):
        bx = (cx > obs[k, 0]) & (cx < obs[k, 2])
        by = (cy > obs[k, 1]) & (cy < obs[k, 3])
        blocked[np.ix_(bx, by)] = True

    # ---- summed-area table for O(1) "is this block of cells free?" ----
    S = np.zeros((nx + 1, ny + 1))
    S[1:, 1:] = blocked.astype(float).cumsum(axis=0).cumsum(axis=1)

    def is_free(i1, i2, j1, j2):
        """Cells i1..i2 x j1..j2 inclusive, 0-based, contain no obstacle."""
        return (S[i2 + 1, j2 + 1] - S[i1, j2 + 1]
                - S[i2 + 1, j1] + S[i1, j1]) == 0

    # ---- maximal free rectangles --------------------------------------
    cand = []
    for i1 in range(nx):
        for i2 in range(i1, nx):
            for j1 in range(ny):
                for j2 in range(j1, ny):
                    if not is_free(i1, i2, j1, j2):
                        break                     # widening j2 cannot help
                    if i1 > 0 and is_free(i1 - 1, i2, j1, j2):
                        continue                  # can grow left  -> not maximal
                    if i2 < nx - 1 and is_free(i1, i2 + 1, j1, j2):
                        continue                  # can grow right
                    if j1 > 0 and is_free(i1, i2, j1 - 1, j2):
                        continue                  # can grow down
                    if j2 < ny - 1 and is_free(i1, i2, j1, j2 + 1):
                        continue                  # can grow up
                    cand.append([i1, i2, j1, j2])

    cand = np.array(cand, dtype=int).reshape(-1, 4)
    if len(cand) == 0:
        raise ValueError("No free rectangle found inside the box.")

    R = np.column_stack([xs[cand[:, 0]], xs[cand[:, 1] + 1],
                         ys[cand[:, 2]], ys[cand[:, 3] + 1]])

    # ---- Lmin filter, then sort widest x-edge first --------------------
    keep = (R[:, 1] - R[:, 0]) >= Lmin - tol
    R, cand = R[keep], cand[keep]

    xspan = R[:, 1] - R[:, 0]
    order = np.argsort(-xspan, kind="stable")     # stable, like MATLAB's sort
    R, cand, xspan = R[order], cand[order], xspan[order]

    all_rects = R
    n = len(R)
    if n == 0:
        raise ValueError(
            f"No free rectangle has x-edge >= Lmin = {Lmin:g}.")

    # ---- greedy cover, widest x-edge first -----------------------------
    uncov = ~blocked
    chosen = []
    while uncov.any():
        best, best_span, best_new = -1, -np.inf, 0
        for r in range(n):
            i1, i2, j1, j2 = cand[r]
            nnew = np.count_nonzero(uncov[i1:i2 + 1, j1:j2 + 1])
            if nnew == 0:
                continue
            if (xspan[r] > best_span + tol
                    or (abs(xspan[r] - best_span) <= tol and nnew > best_new)):
                best, best_span, best_new = r, xspan[r], nnew
        if best < 0:
            warnings.warn(
                f"Lmin = {Lmin:g} leaves part of the free region uncovered.")
            break
        i1, i2, j1, j2 = cand[best]
        uncov[i1:i2 + 1, j1:j2 + 1] = False
        chosen.append(best)

    cover = R[chosen] if chosen else np.zeros((0, 4))

    # ---- overlap graph --------------------------------------------------
    A = np.zeros((n, n), dtype=bool)
    for a in range(n - 1):
        for b in range(a + 1, n):
            ox = min(R[a, 1], R[b, 1]) - max(R[a, 0], R[b, 0])
            oy = min(R[a, 3], R[b, 3]) - max(R[a, 2], R[b, 2])
            if ox >= -tol and oy >= -tol and (ox > tol or oy > tol):
                A[a, b] = A[b, a] = True

    # ---- BFS start -> goal ----------------------------------------------
    def in_R(p):
        return ((p[0] >= R[:, 0] - tol) & (p[0] <= R[:, 1] + tol)
                & (p[1] >= R[:, 2] - tol) & (p[1] <= R[:, 3] + tol))

    src = np.flatnonzero(in_R(p_start))
    tgt = np.flatnonzero(in_R(p_goal))
    if src.size == 0 or tgt.size == 0:
        raise ValueError("Start or goal is not inside any admissible rectangle.")

    dist = np.full(n, np.inf)
    prev = np.full(n, -1, dtype=int)
    dist[src] = 1.0
    queue = list(src)
    while queue:
        a = queue.pop(0)
        nb = np.flatnonzero(A[a] & np.isinf(dist))
        dist[nb] = dist[a] + 1
        prev[nb] = a
        queue.extend(nb.tolist())

    k = int(np.argmin(dist[tgt]))                 # first minimum, as in MATLAB
    if np.isinf(dist[tgt[k]]):
        raise ValueError("No rectangle chain connects start to goal.")

    idx = tgt[k]
    chain = [idx]
    while prev[idx] != -1:
        idx = prev[idx]
        chain.append(idx)

    path_rects = R[chain[::-1]]
    return path_rects, cover, all_rects


if __name__ == "__main__":
   # box = [0, 0, 60, 16]
    #obs = [[26, 2, 32, 10],
      #     [42, 8, 50, 12]]
   # p0, p1 = [0, 12], [58, 6]
   # Lmin = 20

    path_rects, cover, all_rects = max_x_path(box, obs, p0, p1, Lmin)

    np.set_printoptions(suppress=True)
    print("all_rects  (x-edge >= Lmin), widest first:\n", all_rects)
    print("\ncover:\n", cover)
    print("\npath_rects:\n", path_rects)